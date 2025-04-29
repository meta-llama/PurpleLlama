#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this script is meant to run inside of LLDB. Check this doc for more context
# https://lldb.llvm.org/python_api/lldb.SBDebugger.html#lldb.SBDebugger

# load in lldb with:
#   command script import ../microsnapshot.py

import json
import math
import string
import struct
import sys
import traceback
from typing import Any, Dict, List, Optional

import lldb

FAIL_LOUDLY = False  # disabled for eval

SBValue = Any  # lldb.SBValue
SBAddress = Any  # lldb.SBAddress
SBType = Any  # lldb.SBType

IS_PRINTABLE_LUT = [(chr(i) in string.printable) for i in range(0xFF)]

LLDB_CLASSES_TO_NAMES = {
    lldb.eTypeClassAny: "Any",
    lldb.eTypeClassArray: "Array",
    lldb.eTypeClassBlockPointer: "BlockPointer",
    lldb.eTypeClassBuiltin: "Builtin",
    lldb.eTypeClassClass: "Class",
    lldb.eTypeClassComplexFloat: "ComplexFloat",
    lldb.eTypeClassComplexInteger: "ComplexInteger",
    # lldb.eTypeClassFloat: "Float",
    lldb.eTypeClassEnumeration: "Enumeration",
    lldb.eTypeClassFunction: "Function",
    lldb.eTypeClassInvalid: "Invalid",
    lldb.eTypeClassMemberPointer: "Pointer",
    lldb.eTypeClassObjCInterface: "ObjCInterface",
    lldb.eTypeClassObjCObject: "ObjCObject",
    lldb.eTypeClassObjCObjectPointer: "eTypeClassObjCObjectPointer",
    lldb.eTypeClassOther: "Other",
    lldb.eTypeClassPointer: "Pointer",
    lldb.eTypeClassReference: "Reference",
    lldb.eTypeClassStruct: "Struct",
    lldb.eTypeClassTypedef: "Typedef",
    lldb.eTypeClassUnion: "Union",
    lldb.eTypeClassVector: "Vector",
    lldb.eTypeClassInvalid: "Invalid",
}

LLDB_SIGNED_INTS = {
    lldb.eBasicTypeSignedChar,
    lldb.eBasicTypeSignedWChar,
    lldb.eBasicTypeChar16,
    lldb.eBasicTypeChar32,
    lldb.eBasicTypeShort,
    lldb.eBasicTypeInt,
    lldb.eBasicTypeLong,
    lldb.eBasicTypeLongLong,
    lldb.eBasicTypeInt128,
}

LLDB_INTS = {
    lldb.eBasicTypeChar,
    lldb.eBasicTypeSignedChar,
    lldb.eBasicTypeUnsignedChar,
    lldb.eBasicTypeWChar,
    lldb.eBasicTypeSignedWChar,
    lldb.eBasicTypeUnsignedWChar,
    lldb.eBasicTypeChar16,
    lldb.eBasicTypeChar32,
    lldb.eBasicTypeShort,
    lldb.eBasicTypeUnsignedShort,
    lldb.eBasicTypeInt,
    lldb.eBasicTypeUnsignedInt,
    lldb.eBasicTypeLong,
    lldb.eBasicTypeUnsignedLong,
    lldb.eBasicTypeLongLong,
    lldb.eBasicTypeUnsignedLongLong,
    lldb.eBasicTypeInt128,
    lldb.eBasicTypeUnsignedInt128,
    lldb.eBasicTypeBool,
}

LLDB_FLOATS = {
    lldb.eBasicTypeHalf,
    lldb.eBasicTypeFloat,
    lldb.eBasicTypeDouble,
    lldb.eBasicTypeLongDouble,
}


def is_function(tt):
    while tt.GetTypeClass() == lldb.eTypeClassTypedef:
        tt = tt.GetTypedefedType()
    return tt.GetTypeClass() == lldb.eTypeClassFunction


def is_signed(tt):
    return tt.GetBasicType() in LLDB_SIGNED_INTS


def is_int(tt):
    return tt.GetBasicType() in LLDB_INTS


def is_float(tt):
    return tt.GetBasicType() in LLDB_FLOATS


def escape_str(bytes_array):
    res = ""
    for c in bytes_array:
        if chr(c) == '"':
            res += "\\x22"
        elif chr(c) == "\\":
            res += "\\\\"
        elif IS_PRINTABLE_LUT[c]:
            res += chr(c)
        else:
            return None
    return res


def strip_prefix(prefix, string):
    if string.startswith(prefix):
        return string[len(prefix) :]
    return string


def strip_const(type_name):
    return strip_prefix("const", type_name)


class BackRef:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"BACKREF({self.name})"

    def as_json(self):
        return {"stype": "BACKREF", "name": self.name}


class Alloc:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"ALLOC({self.name}, {self.value})"

    def as_json(self):
        return {"stype": "ALLOC", "name": self.name, "value": self.value}


class FuncPtr:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"FPTR({self.name})"

    def as_json(self):
        return {"stype": "FPTR", "name": self.name}


class DumpError:
    def __init__(self, desc: str):
        self.desc = desc

    def __repr__(self):
        return f"ERROR({self.desc})"

    def as_json(self):
        return {"stype": "ERROR", "desc": self.desc}


class Config:
    def __init__(self, json_val: Dict[str, Any], outfile: str):
        self.struct_to_skipped_fields = {}
        for struct_name in json_val["struct_to_skipped_fields"]:
            self.struct_to_skipped_fields[struct_name] = set(
                json_val["struct_to_skipped_fields"][struct_name]
            )
        self.skipped_ptr_types = json_val["skipped_ptr_types"]
        self.skipped_locals = set(json_val["skipped_locals"])
        self.outfile_path = outfile
        self.breakpoint_to_expressions = json_val["breakpoint_to_expressions"]
        self.dump_after_return_breakpoints = json_val["dump_after_return_breakpoints"]
        self.max_objects = json_val["max_objects"]
        self.max_recursion_depth = json_val["max_recursion_depth"]
        self.catch_exceptions = json_val["catch_exceptions"]
        self.dump_stacktrace_breakpoints = json_val["dump_stacktrace_breakpoints"]

    def skip_field(self, struct_name: str, field_name: str) -> bool:
        if struct_name in self.struct_to_skipped_fields:
            return field_name in self.struct_to_skipped_fields[struct_name]
        return False


class RecursiveDumpValues:
    def __init__(self, debugger, config: Config):
        self.debugger = debugger
        self.type_to_alloc = {}
        self.code = ""
        self.ptrs_to_names = {}
        self.ptr_counter = 0
        self.count_objects = 0
        self.recursion_depth = 0
        self.config = config
        self.ptr_type_handlers = {
            "const char": self.type_maybe_str,
            "char": self.type_maybe_str,
        }
        self.string_cache = {}
        self.type_allocators = {}

    def load_string(self, val):
        if val.load_addr in self.string_cache:
            return self.string_cache[val.load_addr]
        try:
            mem_bytes = get_val_mem(val, 128)
            if not mem_bytes:
                return None
            last = mem_bytes.index(0)
            string = escape_str(mem_bytes[:last])
            self.string_cache[val.load_addr] = string
            return string
        except Exception:
            e = sys.exc_info()[0]
            self.log(f"couldn't generate string for {val}: {e}")
        return None

    def type_maybe_str(self, val):
        string = self.load_string(val)
        if string:
            return string
        return self.dump_pointer(val)

    def type_skip(self, mem, tt, src_expr):
        self.log(f"skipped {src_expr} (type: {tt.name})")
        pass

    def get_ptr_name(self, tt):
        self.ptr_counter += 1
        return f"ptr_{self.ptr_counter}"

    def start_step(self):
        self.count_objects += 1
        if self.count_objects > self.config.max_objects:
            raise Exception(
                f"Trying to dump too many objects (max: {self.config.max_objects})"
            )
        self.recursion_depth += 1
        if self.recursion_depth > self.config.max_recursion_depth:
            raise Exception(
                f"Trying recurse too deep (max: {self.config.max_recursion_depth})"
            )

    def finish_step(self):
        self.recursion_depth -= 1

    def log(self, str):
        print(f"[INFO]: {str}")

    def dump(self, val: Optional[SBValue] = None, tt: Optional[SBType] = None):
        # TODO: Handle C++ std smart pointers
        # TODO: Handle large arrays (never dump more than the first couple 100 values?)
        # TODO: Make it easy to calculate the length for pointer types
        try:
            self.start_step()
            res = self.dump_inner(val, tt)
        except Exception as e:
            if self.config.catch_exceptions:
                too_deep = "Trying recurse too deep " in str(e)
                too_many = "Trying to dump too many object" in str(e)
                if not too_deep and not too_many:
                    print("caught exception", e)
                    print("handle exception: " + traceback.format_exc())
                if FAIL_LOUDLY:
                    res = DumpError(f"Error while dumping {val}: {e}")
                else:
                    res = DumpError(f"Error while dumping")
            else:
                raise e

        finally:
            self.finish_step()
        return res

    def dump_inner(self, val: Optional[SBValue] = None, tt: Optional[SBType] = None):
        if tt is None:
            tt = val.type
        type_class = tt.GetTypeClass()
        if type_class == lldb.eTypeClassTypedef:
            return self.dump(val=val, tt=tt.GetTypedefedType())
        elif type_class == lldb.eTypeClassArray:
            return self.dump_array(val)
        elif type_class == lldb.eTypeClassBuiltin:
            return self.dump_builtin(val)
        elif type_class == lldb.eTypeClassClass:
            return self.dump_class(val)
        elif type_class == lldb.eTypeClassStruct:
            return self.dump_class(val)
        elif type_class == lldb.eTypeClassPointer:
            pointee = tt.GetPointeeType()
            if is_function(pointee):
                return self.dump_fptr(val)
            elif pointee.name == "void":
                return self.dump_voidptr(val)
            elif pointee.name in self.ptr_type_handlers:
                return self.ptr_type_handlers[pointee.name](val)
            return self.dump_pointer(val)
        elif type_class == lldb.eTypeClassEnumeration:
            return self.dump_enum(val)
        elif type_class == lldb.eTypeClassUnion:
            return self.dump_union(val)
        else:
            msg = f"unhandled TypeClass {LLDB_CLASSES_TO_NAMES.get(type_class, str(type_class))} on {tt.name} with for {val}"
            if FAIL_LOUDLY:
                raise BaseException(msg)
            else:
                print("ERROR: " + msg)
                return None

    def dump_std_string(self, val):
        return val.GetSummary()

    def dump_class(self, val):
        tt = val.type
        class_name = strip_const(tt.GetDisplayTypeName().split("<")[0]).strip()
        # print(f"generate for class {val} with name {class_name}")
        # using the synthetic provider for std::vector defined here:
        # https://spclgitlab.ethz.ch/open-climate-compiler/llvm-project/-/blob/master/lldb/examples/synthetic/gnu_libstdcpp.py?ref_type=heads
        if class_name == "std::vector":
            return self.dump_array(val)
        if class_name == "std::map":
            return self.dump_array(val)
        if class_name == "std::basic_string" or class_name == "std::string":
            return self.dump_std_string(val)
        if class_name == "std::pair":
            return self.dump_array(val)
        return self.dump_struct(val)

    def dump_struct(self, sbval):
        tt = sbval.type
        res = {"stype": "STRUCT", "type": tt.name, "values": {}}
        for i in range(sbval.type.GetNumberOfFields()):
            child_val = sbval.GetChildAtIndex(i)
            field = sbval.type.GetFieldAtIndex(i)
            if not self.config.skip_field(tt.name, field.name):
                val = self.dump(val=child_val)
                res["values"][field.name] = val
        return res

    def dump_voidptr(self, val):
        return "void-ptr"

    def resolve_fptr_name(self, val):
        if isinstance(val, lldb.SBAddress):
            return val.symbol.name or "func-ptr"
        return "func-ptr"

    def dump_fptr(self, val):
        return FuncPtr(self.resolve_fptr_name(val))

    def dump_enum(self, val):
        return val.value

    def dump_pointer(self, val):
        addr = val.Dereference().load_addr
        tt = val.type
        if addr == 0:
            return None
        if addr in self.ptrs_to_names:
            return BackRef(self.ptrs_to_names[addr])
        name = self.get_ptr_name(tt)
        self.ptrs_to_names[addr] = name
        res = self.dump(val=val.Dereference())
        return Alloc(name, res)

    def dump_builtin(self, val):
        error_ref = lldb.SBError()
        tt = val.type
        while tt.GetTypeClass() == lldb.eTypeClassTypedef:
            tt = tt.GetTypedefedType()
        if is_int(tt):
            if is_signed(tt):
                intval = val.GetValueAsSigned(error_ref)
            else:
                intval = val.GetValueAsUnsigned(error_ref)
            if not error_ref.Success():
                self.log(f"// couldn't get value for {val} ({tt.name}) {error_ref}")
            return intval
        elif is_float(tt):
            val = round(float(val.value), 5)
            if math.isnan(val):
                val = "NAN"
        else:
            raise BaseException(f"Unknown Builting type {tt.name} for {val} {repr(tt)}")
        return val

    def dump_array(self, val):
        res = []
        for i in range(min([val.num_children, 128])):
            res.append(
                self.dump(
                    val=val.GetChildAtIndex(i),
                )
            )
        return res

    def dump_union(self, val):
        return {"stype": "UNION", "type": val.type.name}

    def dump_expr(self, expr) -> Dict[str, Any]:
        target = self.debugger.GetSelectedTarget()
        val = target.EvaluateExpression(expr)
        return self.dump(val=val)

    def dump_to_file(self, expr, outfile):
        res = self.dump_expr(expr)
        strval = json.dumps(res, default=lambda o: o.as_json())
        print(strval)
        with open(outfile, "w") as f:
            f.write(strval)


def get_val_mem(val: SBValue, size: Optional[int] = None):
    error_ref = lldb.SBError()
    addr = val.GetData().GetAddress(error_ref, 0)
    if not error_ref.Success():
        return None
    if size is None:
        size = val.GetData().GetByteSize()
    if (addr <= 0) or (size <= 0):
        return None
    memory = val.process.ReadMemory(addr, size, error_ref)
    if error_ref.Success():
        return memory
    return None


class MarkerObj:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}:{id(self)}>"


class GlobalState:
    config: Config
    address_to_expressions: Dict[str, List[str]]

    def __init__(self):
        self.config = None
        self.address_to_expressions = {}
        self.address_to_desc = {}
        # just a sentinel value to indicate that we want to place a hook after returning from the current function
        self.EXPR_HOOK_AFTER_RETURN = MarkerObj("hook_after_return")
        # just a sentinel value to indicate that we want to dump the whole frame if we hit this breakpoint
        self.EXPR_DUMP_FRAME = MarkerObj("dump_frame")
        # dump the stracktrace to identify parent functions that will make good breakpoint locations
        self.EXPR_DUMP_STACKTRACE = MarkerObj("dump_stacktrace")


GLOBAL_STATE = GlobalState()


def key_for_loc(loc: SBAddress):
    return f"{loc.GetModule().GetFileSpec().basename}+{hex(loc.GetFileAddress())}"


def setup_bp(bp, desc, exprs):
    global GLOBAL_STATE
    assert bp.IsValid()
    assert bp.enabled
    bp.SetAutoContinue(True)
    for i in range(bp.GetNumLocations()):
        loc = bp.GetLocationAtIndex(i)
        addr = key_for_loc(loc.GetAddress())
        GLOBAL_STATE.address_to_desc[addr] = desc
        GLOBAL_STATE.address_to_expressions[addr] = exprs
        bp.SetScriptCallbackFunction("microsnapshot.dump_struct_callback_wrapper")


def add_bp(target, desc, exprs, addr=None):
    global GLOBAL_STATE
    if addr:
        assert isinstance(addr, lldb.SBAddress)
        if key_for_loc(addr) in GLOBAL_STATE.address_to_expressions:
            return
        bp = target.BreakpointCreateByAddress(addr.GetLoadAddress(target))
    elif isinstance(desc, str):
        if desc.count(":") == 1:
            file, linenum = desc.split(":")
            bp = target.BreakpointCreateByLocation(file, int(linenum))
        else:
            bp = target.BreakpointCreateByName(desc)
    else:
        print(f"FAILED ADDING BP FOR {desc} {type(desc)}")
        return
    # assert bp.GetNumLocations() == 1
    setup_bp(bp, desc, exprs)


def dump_struct_callback_wrapper(frame, loc, dict):
    global GLOBAL_STATE
    thread = frame.GetThread()
    process = thread.GetProcess()
    target = process.GetTarget()
    debugger = target.GetDebugger()
    addr = key_for_loc(loc.GetAddress())
    if addr not in GLOBAL_STATE.address_to_expressions:
        print(
            f"no expressions for {repr(addr)}! registered: {repr(list(GLOBAL_STATE.address_to_expressions.keys()))}"
        )
    exprs = GLOBAL_STATE.address_to_expressions[addr]
    desc = GLOBAL_STATE.address_to_desc[addr]
    res = None
    print(f"hit breakpoint at {loc} -> desc: {desc} -> exprs: {exprs}")
    if exprs == GLOBAL_STATE.EXPR_HOOK_AFTER_RETURN:
        parent = frame.parent
        # When we hit a breakpoint of typ "HOOK AFTER RETURN", we want to place
        # a breakpoint after the current function returns - that breakpoint will
        # then dump the whole frame of our current function's parent
        if key_for_loc(parent.addr) not in GLOBAL_STATE.address_to_expressions:
            add_bp(target, f"after({desc})", GLOBAL_STATE.EXPR_DUMP_FRAME, parent.addr)
    elif exprs == GLOBAL_STATE.EXPR_DUMP_FRAME:
        res = {"desc": desc, "values": {}}
        dumper = RecursiveDumpValues(debugger, GLOBAL_STATE.config)
        for val in frame.vars:
            if val.name not in GLOBAL_STATE.config.skipped_locals:
                res["values"][val.name] = dumper.dump(val)
    elif exprs == GLOBAL_STATE.EXPR_DUMP_STACKTRACE:
        res = []
        for _ in range(10000):
            res.append(frame.name)
            if "LLVMFuzzerTestOneInput" in frame.name:
                break
            frame = frame.parent
            if not frame:
                break
    else:
        res = {"desc": desc, "values": {}}
        dumper = RecursiveDumpValues(debugger, GLOBAL_STATE.config)
        for expr in exprs:
            # We reuse the dumper, so that we don't dump the same pointers multiple times if multiple expressions refer to the same object
            res["values"][expr] = dumper.dump_expr(expr)
    if res:
        with open(GLOBAL_STATE.config.outfile_path, "a") as f:
            json.dump(res, f, default=lambda o: o.as_json())
            f.write("\n")


def cmd_dump_struct_expr(debugger, command, result, internal_dict):
    expr, rules_file, outfile = command.split(" ")
    print("dump struct on: " + repr((expr, rules_file, outfile)))
    with open(rules_file, "r") as f:
        rules = Config(json.load(f), outfile)
    RecursiveDumpValues(debugger, rules).dump_to_file(expr, outfile)


def cmd_hook_dump_structs(debugger, command, result, internal_dict):
    global GLOBAL_STATE
    rules_file, outfile = command.split(" ")
    with open(rules_file, "r") as f:
        config = Config(json.load(f), outfile)

    GLOBAL_STATE.config = config
    with open(GLOBAL_STATE.config.outfile_path, "w+") as f:
        pass
    target = debugger.GetSelectedTarget()
    for desc, exprs in config.breakpoint_to_expressions.items():
        add_bp(target, desc, exprs)
    for desc in config.dump_after_return_breakpoints:
        add_bp(target, desc, GLOBAL_STATE.EXPR_HOOK_AFTER_RETURN)


def cmd_hook_dump_stacktrace(debugger, command, result, internal_dict):
    print("RUNNING HOOK DUMP STACKTRACE")
    global GLOBAL_STATE
    rules_file, outfile = command.split(" ")
    with open(rules_file, "r") as f:
        config = Config(json.load(f), outfile)

    GLOBAL_STATE.config = config
    with open(GLOBAL_STATE.config.outfile_path, "w+") as f:
        pass
    target = debugger.GetSelectedTarget()
    for desc in config.dump_stacktrace_breakpoints:
        print(f"ADDING BP FOR {desc}")
        add_bp(target, desc, GLOBAL_STATE.EXPR_DUMP_STACKTRACE)


# And the initialization code to add your commands
def __lldb_init_module(debugger, internal_dict):
    # lldb doesn't autoamtically reload dependencies if we reload this script, lets do it manually
    debugger.HandleCommand(
        "command script add -f microsnapshot.cmd_dump_struct_expr dump_struct"
    )
    debugger.HandleCommand(
        "command script add -f microsnapshot.cmd_hook_dump_structs hook_dump_structs"
    )
    debugger.HandleCommand(
        "command script add -f microsnapshot.cmd_hook_dump_stacktrace hook_dump_stacktrace"
    )
    print(
        'The "dump_struct" and "run_dump_structs", "hook_dump_stacktrace" python command have been installed and is ready for use.'
    )
