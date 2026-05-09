
current=$(pwd)

cd extern/phydll

mkdir -p ./build
export BUILD=$(realpath ./build)
PHYDLL_CC=${PHYDLL_CC:-${MPICC:-}}
if [ -z "$PHYDLL_CC" ]; then
    if command -v mpicc >/dev/null 2>&1; then
        PHYDLL_CC=mpicc
    else
        PHYDLL_CC=${CC:-gcc}
    fi
fi

echo "$PHYDLL_CC"
MPI_LDFLAGS=""
if "$PHYDLL_CC" --showme:link >/dev/null 2>&1; then
    MPI_LDFLAGS=$($PHYDLL_CC --showme:link)
fi

if [[ -z "${NP_PHY:-}" && -z "${NP_DL:-}" && -z "${NP_TOTAL:-}" ]]; then
    export NP_PHY=1
    export NP_DL=1
fi

make CC="$PHYDLL_CC" BUILD="$BUILD" MPI_LDFLAGS="$MPI_LDFLAGS"

PHYDLL_TEST_TARGET=${PHYDLL_TEST_TARGET:-crun}
PHYDLL_ENABLE_FORTRAN=${PHYDLL_ENABLE_FORTRAN:-ON}
PHYDLL_ENABLE_PYTHON=${PHYDLL_ENABLE_PYTHON:-ON}
make CC="$PHYDLL_CC" BUILD="$BUILD" MPI_LDFLAGS="$MPI_LDFLAGS" \
    ENABLE_FORTRAN="$PHYDLL_ENABLE_FORTRAN" ENABLE_PYTHON="$PHYDLL_ENABLE_PYTHON" \
    "$PHYDLL_TEST_TARGET"

ls ./build


cd "$current"
