AC_DEFUN([AX_PATH_CULA],
[
    AC_PROVIDE([AX_PATH_CULA])

    AC_MSG_CHECKING([whether CULA is installed])

    CULA_CPPFLAGS=""
    if test -n "$CULA_INC_PATH"; then
        CULA_CPPFLAGS="-I$CULA_INC_PATH"
    fi

    CULA_LIBS=""
    if test -n "$CULA_LIB_PATH_64"; then
        CULA_LIBS="$CULA_LIBS -L$CULA_LIB_PATH_64"
    elif test -n "$CULA_LIB_PATH_32"; then
        CULA_LIBS="$CULA_LIBS -L$CULA_LIB_PATH_32"
    fi
    if test -n "$CULA_LIBS"; then
        CULA_LIBS="$CULA_LIBS -lcula_lapack"
    fi

    AC_LANG_PUSH(C++)
    ac_save_CPPFLAGS="$CPPFLAGS"
    ac_save_LIBS="$LIBS"
    CPPFLAGS="$ac_save_CPPFLAGS $CULA_CPPFLAGS"
    LIBS="$ac_save_LIBS $CULA_LIBS"
    AC_TRY_LINK(
        [#include <cula.h>],
        [culaStatus status = culaInitialize();],
        have_cula=yes, have_cula=no
    )
    LIBS="$ac_save_LIBS"
    CPPFLAGS="$ac_save_CPPFLAGS"
    AC_LANG_POP(C++)

    AC_MSG_RESULT($have_cula)

    if test "$have_cula" = "no"; then
        CULA_CPPFLAGS=""
        CULA_LIBS=""
    fi
])
