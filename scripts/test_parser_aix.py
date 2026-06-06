import clang.cindex
import sys
import os

# Prepend libclang path if available
libclang_path = os.environ.get('LIBCLANG_PATH', '/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Clang/15.0.5-GCCcore-11.3.0/lib')
if libclang_path:
    if os.path.isdir(libclang_path):
        # Find the actual library file
        found = False
        for f in sorted(os.listdir(libclang_path)):
            if f.startswith('libclang.so') and not f.endswith('.a'):
                clang.cindex.Config.set_library_file(os.path.join(libclang_path, f))
                found = True
                break
        if not found:
            print(f"Warning: libclang.so not found in {libclang_path}")
    else:
        clang.cindex.Config.set_library_file(libclang_path)

def test_parse(header_path):
    index = clang.cindex.Index.create()
    args = ['-std=c++17', '-I.', '-Iinclude', '-xc++']
    
    tu = index.parse(header_path, args=args)
    
    def find_class(node, name):
        if (node.kind == clang.cindex.CursorKind.CLASS_TEMPLATE or node.kind == clang.cindex.CursorKind.CLASS_DECL) and node.spelling == name:
            return node
        for child in node.get_children():
            res = find_class(child, name)
            if res: return res
        return None

    cls = find_class(tu.cursor, "MLCouplingProviderAixelerator")
    if not cls:
        print("Class not found")
        return

    for child in cls.get_children():
        if child.kind == clang.cindex.CursorKind.CONSTRUCTOR:
            print(f"Constructor: {child.spelling}")
            params = [p for p in child.get_children() if p.kind == clang.cindex.CursorKind.PARM_DECL]
            print(f"  Num params: {len(params)}")
            for p in params:
                print(f"    Param: {p.spelling}, Type: {p.type.spelling}")
            
            tokens = [t.spelling for t in child.get_tokens()]
            print(f"  Tokens: {' '.join(tokens)}")

if __name__ == "__main__":
    header = "include/provider/ml_coupling_provider_aixelerator.hpp"
    # Ensure we are in the right directory
    os.chdir("/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface")
    test_parse(header)
