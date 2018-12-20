m4_include([tools/gsl.m4])
m4_include([tools/cula.m4])

AC_INIT([R2GUESS], [2.0])

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
    AC_MSG_ERROR([Could not determine R_HOME])
fi

CXX=`"${R_HOME}/bin/R" CMD config CXX`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`

AC_LANG([C++])

AX_PATH_GSL([1.12], [],
            AC_MSG_ERROR([could not find required version of GSL]))
CPPFLAGS="$CPPFLAGS $GSL_CFLAGS"
LDFLAGS="$LDFLAGS $GSL_LIBS"

AX_PATH_CULA
if test "$have_cula" == "yes"; then
    AC_MSG_NOTICE([GPU acceleration will be enabled])
    CPPFLAGS="$CPPFLAGS $CULA_CPPFLAGS -D_CUDA_"
    LDFLAGS="$LDFLAGS $CULA_LIBS"
fi

AC_CONFIG_FILES(src/Makevars)

AC_OUTPUT

