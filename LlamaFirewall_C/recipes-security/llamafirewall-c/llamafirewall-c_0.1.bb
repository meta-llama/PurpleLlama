# LlamaFirewall_C/recipes-security/llamafirewall-c/llamafirewall-c_0.1.bb
SUMMARY = "LlamaFirewall C D-Bus Service"
DESCRIPTION = "A C implementation of LlamaFirewall components, exposed via D-Bus."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=24e3e330db6e9998995a200659bc0a7c"

DEPENDS = "pcre2 systemd"

# This recipe assumes the LlamaFirewall_C directory (which contains src/, conf/, LICENSE,
# and this recipes-security/ folder) is the root of the source for this recipe.
SRC_URI = "file://./"

# The dummy python script is part of the main source now.
# It will be found in ${S}/src/promptguard_helper.py after unpacking.

# S is the source directory, which is ${WORKDIR} when SRC_URI is "file://./"
S = "${WORKDIR}"

# B is the build directory, relative to S. Our Makefile is in S/src.
B = "${S}/src"

do_compile() {
    oe_runmake -C ${B} all CC="${CC}" CFLAGS="${CFLAGS}" LDFLAGS="${LDFLAGS}"
}

do_install() {
    # Install the executable
    install -d ${D}${bindir}
    install -m 0755 ${B}/llamafirewall-dbus-service ${D}${bindir}/

    # Install the D-Bus service file from S/conf
    install -d ${D}${datadir}/dbus-1/system-services
    install -m 0644 ${S}/conf/io.LlamaFirewall.service ${D}${datadir}/dbus-1/system-services/

    # Install the Python helper script from S/src
    install -d ${D}${nonarch_base_libdir}/llamafirewall-c
    install -m 0755 ${S}/src/promptguard_helper.py ${D}${nonarch_base_libdir}/llamafirewall-c/
}

FILES:${PN} += "${datadir}/dbus-1/system-services/io.LlamaFirewall.service"
FILES:${PN} += "${nonarch_base_libdir}/llamafirewall-c/promptguard_helper.py"

RDEPENDS:${PN} += "pcre2 python3"
