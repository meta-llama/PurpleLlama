#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h> // For ENOMEM etc.
#include <systemd/sd-bus.h> // sd-bus library

#include "llamafirewall_core.h"
#include "llamafirewall_types.h"

#define DBUS_SERVICE_NAME "io.LlamaFirewall"
#define DBUS_OBJECT_PATH "/io/LlamaFirewall"
#define DBUS_INTERFACE_NAME "io.LlamaFirewall.Service1"

static LlamaFirewallCore firewall_core;
static int core_initialized = 0;

// Forward declaration
static int method_scan_message(sd_bus_message *m, void *userdata, sd_bus_error *ret_error);

// Method table for the D-Bus interface
static const sd_bus_vtable llamafirewall_vtable[] = {
    SD_BUS_VTABLE_START(0),
    SD_BUS_METHOD("ScanMessage", "ssas", "ssds", method_scan_message, SD_BUS_VTABLE_UNPRIVILEGED),
    // "ssas": input signature: string (content), string (role), array of structs (string, string) for trace
    // "ssds": output signature: string (decision), string (reason), double (score), string (status)
    SD_BUS_VTABLE_END
};

// Helper to initialize the core if not already done.
static int ensure_core_initialized(sd_bus_error *ret_error) {
    if (!core_initialized) {
        if (llamafirewall_core_init(&firewall_core) != 0) {
            sd_bus_error_set_const(ret_error, "io.LlamaFirewall.Error.CoreInitFailed", "Failed to initialize LlamaFirewall core.");
            return -1;
        }
        core_initialized = 1;
    }
    return 0;
}


static int method_scan_message(sd_bus_message *m, void *userdata, sd_bus_error *ret_error) {
    const char *message_content_dbus;
    const char *message_role_dbus;
    int r;

    if (ensure_core_initialized(ret_error) < 0) {
        // error is already set by ensure_core_initialized
        return sd_bus_reply_method_errorf(m, ret_error->name, ret_error->message);
    }

    r = sd_bus_message_read_basic(m, 's', &message_content_dbus);
    if (r < 0) {
        sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_INVALID_ARGS, "Failed to parse message_content: %s", strerror(-r));
        return r;
    }
    r = sd_bus_message_read_basic(m, 's', &message_role_dbus);
    if (r < 0) {
        sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_INVALID_ARGS, "Failed to parse message_role: %s", strerror(-r));
        return r;
    }

    CMessage current_message;
    current_message.role = c_role_from_string(message_role_dbus);
    current_message.content = strdup(message_content_dbus);
    if (!current_message.content) {
        sd_bus_error_set_const(ret_error, SD_BUS_ERROR_NO_MEMORY, "Failed to allocate memory for message content.");
        return -ENOMEM;
    }


    if (current_message.role == ROLE_UNKNOWN && strcmp(message_role_dbus, c_role_to_string(ROLE_UNKNOWN)) != 0) {
        free(current_message.content);
        sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_INVALID_ARGS, "Invalid message_role: %s", message_role_dbus);
        return sd_bus_reply_method_errorf(m, ret_error->name, ret_error->message);
    }

    CTrace current_trace = { .messages = NULL, .count = 0 };
    CMessage* temp_trace_messages = NULL;
    size_t temp_trace_capacity = 0;

    r = sd_bus_message_enter_container(m, 'a', "(ss)");
    if (r < 0) {
        free(current_message.content);
        sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_INVALID_ARGS, "Failed to enter trace_history array container: %s", strerror(-r));
        return r;
    }

    while ((r = sd_bus_message_enter_container(m, 'r', "ss")) > 0) { // Enter struct
        const char *trace_role_dbus_item, *trace_content_dbus_item;
        r = sd_bus_message_read_basic(m, 's', &trace_role_dbus_item);
        if (r < 0) { sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_INVALID_ARGS, "Failed to read trace role: %s", strerror(-r)); break; }

        r = sd_bus_message_read_basic(m, 's', &trace_content_dbus_item);
        if (r < 0) { sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_INVALID_ARGS, "Failed to read trace content: %s", strerror(-r)); break; }

        if (current_trace.count >= temp_trace_capacity) {
            temp_trace_capacity = (temp_trace_capacity == 0) ? 4 : temp_trace_capacity * 2;
            CMessage* new_alloc = (CMessage*)realloc(temp_trace_messages, temp_trace_capacity * sizeof(CMessage));
            if (!new_alloc) {
                sd_bus_error_set_const(ret_error, SD_BUS_ERROR_NO_MEMORY, "Failed to reallocate trace messages.");
                r = -ENOMEM; break;
            }
            temp_trace_messages = new_alloc;
        }

        CRole role_item = c_role_from_string(trace_role_dbus_item);
        // Optional: Validate role_item if necessary, e.g., if it's ROLE_UNKNOWN
        // and trace_role_dbus_item is not "unknown_role"

        temp_trace_messages[current_trace.count].role = role_item;
        temp_trace_messages[current_trace.count].content = strdup(trace_content_dbus_item);
        if (!temp_trace_messages[current_trace.count].content) {
            sd_bus_error_set_const(ret_error, SD_BUS_ERROR_NO_MEMORY, "Failed to duplicate trace item content.");
            r = -ENOMEM; break;
        }
        current_trace.count++;
        r = sd_bus_message_exit_container(m); // Exit struct
        if (r < 0) { sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_INVALID_ARGS, "Failed to exit trace struct container: %s", strerror(-r)); break; }
    }
    current_trace.messages = temp_trace_messages;

    // After the loop, r might be < 0 if an error occurred, or r == 0 if sd_bus_message_enter_container for 'r' indicated end of array.
    if (r < 0) { // Error during trace reading loop
        for (size_t i = 0; i < current_trace.count; ++i) free(current_trace.messages[i].content);
        free(current_trace.messages);
        free(current_message.content);
        // ret_error should already be set by the code inside the loop.
        return sd_bus_reply_method_errorf(m, ret_error->name, ret_error->message);
    }

    r = sd_bus_message_exit_container(m); // Exit array
    if (r < 0) {
        for (size_t i = 0; i < current_trace.count; ++i) free(current_trace.messages[i].content);
        free(current_trace.messages);
        free(current_message.content);
        sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_INVALID_ARGS, "Failed to exit trace_history array container: %s", strerror(-r));
        return r;
    }

    CScanResult scan_result = llamafirewall_scan_message(&firewall_core, &current_message, &current_trace);

    free(current_message.content);
    for (size_t i = 0; i < current_trace.count; ++i) {
        free(current_trace.messages[i].content);
    }
    free(current_trace.messages);

    sd_bus_message *reply = NULL;
    r = sd_bus_message_new_method_return(m, &reply);
    if (r < 0) {
        sd_bus_error_set_printf(ret_error, SD_BUS_ERROR_NO_MEMORY, "Failed to create reply message: %s", strerror(-r));
        free_scan_result_members(&scan_result);
        return r;
    }

    const char* decision_str = c_scan_decision_to_string(scan_result.decision);
    const char* status_str = c_scan_status_to_string(scan_result.status);

    if (sd_bus_message_append_basic(reply, 's', decision_str) < 0 ||
        sd_bus_message_append_basic(reply, 's', scan_result.reason ? scan_result.reason : "") < 0 ||
        sd_bus_message_append_basic(reply, 'd', &scan_result.score) < 0 ||
        sd_bus_message_append_basic(reply, 's', status_str) < 0) {
        sd_bus_error_set_const(ret_error, SD_BUS_ERROR_NOMEM, "Failed to append to reply message."); // Simplified error
        free_scan_result_members(&scan_result);
        // sd_bus_message_unref(reply); // Reply is not sent, unref it.
        return -ENOMEM; // Or appropriate error
    }

    free_scan_result_members(&scan_result);
    return sd_bus_send(NULL, reply, NULL);
}

static void core_final_cleanup(void) {
    if (core_initialized) {
        llamafirewall_core_cleanup(&firewall_core);
        core_initialized = 0;
        printf("LlamaFirewallCore cleaned up via atexit/signal.\n");
    }
}

int main(int argc, char *argv[]) {
    sd_bus_slot *slot = NULL;
    sd_bus *bus = NULL;
    int r;

    r = sd_bus_open_system(&bus);
    if (r < 0) {
        fprintf(stderr, "Failed to connect to system bus: %s\n", strerror(-r));
        goto finish;
    }

    r = sd_bus_add_object_vtable(bus, &slot, DBUS_OBJECT_PATH, DBUS_INTERFACE_NAME, llamafirewall_vtable, NULL);
    if (r < 0) {
        fprintf(stderr, "Failed to add object vtable: %s\n", strerror(-r));
        goto finish;
    }

    r = sd_bus_request_name(bus, DBUS_SERVICE_NAME, 0);
    if (r < 0) {
        fprintf(stderr, "Failed to acquire service name '%s': %s\n", DBUS_SERVICE_NAME, strerror(-r));
        goto finish;
    }

    printf("LlamaFirewall D-Bus service started. Listening on %s %s %s\n",
           DBUS_SERVICE_NAME, DBUS_OBJECT_PATH, DBUS_INTERFACE_NAME);
    printf("Press Ctrl+C to exit.\n");

    atexit(core_final_cleanup); // Register cleanup function

    for (;;) {
        r = sd_bus_process(bus, NULL);
        if (r < 0) {
            fprintf(stderr, "Failed to process bus: %s\n", strerror(-r));
            goto finish;
        }
        if (r > 0) continue;

        r = sd_bus_wait(bus, (uint64_t)-1);
        if (r < 0) {
            fprintf(stderr, "Failed to wait for bus: %s\n", strerror(-r));
            goto finish;
        }
    }

finish:
    // core_final_cleanup() will be called by atexit if main loop broken
    // or can be called explicitly if atexit is not used for signals.
    sd_bus_slot_unref(slot);
    sd_bus_unref(bus);

    return r < 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
