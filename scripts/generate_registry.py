#!/usr/bin/env python3
"""
Registry Generator Script

Parses C++ header files using libclang to find base classes and their subclasses,
then generates a registry header file with factory functions and lookup utilities.

Usage:
    python generate_registry.py <output_path> <base_classes> <header_files...>
"""

import sys
import os
import errno
import shlex
import re

try:
    import clang.cindex
except ModuleNotFoundError:
    clang = None


# =============================================================================
# Libclang Initialization
# =============================================================================

def init_libclang():
    """Initialize libclang, trying multiple common paths if needed."""
    if clang is None:
        raise ModuleNotFoundError(
            "The Python 'clang' module is required to generate the registry. "
            "Install it or run the generator in the project environment."
        )
    try:
        return clang.cindex.Index.create()
    except clang.cindex.LibclangError:
        env_path = os.environ.get('LIBCLANG_PATH')
        candidates = []
        if env_path is not None and env_path != "":
            for dirpath, _, filenames in os.walk(env_path):
                for f in filenames:
                    if f.startswith("libclang.so"):
                        candidates.append(os.path.join(dirpath, f))
        else:
            candidates = [
                os.environ.get('LIBCLANG_PATH'),
                '/usr/lib/llvm-18/lib/libclang.so.1',
                '/usr/lib/x86_64-linux-gnu/libclang-18.so.18',
                '/usr/lib/x86_64-linux-gnu/libclang-18.so.1',
                '/usr/lib/llvm-15/lib/libclang.so.1',
                '/usr/lib/llvm-11/lib/libclang.so.1',
            ]
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                clang.cindex.Config.set_library_file(path)
                return clang.cindex.Index.create()
            except OSError as e:
                if getattr(e, 'errno', None) in (errno.ENOENT,):
                    continue
        raise

def get_parse_args():
    """Get clang parse arguments from environment or use defaults."""
    env_args = os.environ.get('CLANG_ARGS')
    if env_args:
        try:
            return shlex.split(env_args)
        except Exception:
            pass
    
    args = ['-std=c++17', '-I.', '-Iinclude', '-xc++']
    for sys_include in ['/usr/include/c++/11', '/usr/include/c++/13', '/usr/include']:
        if os.path.exists(sys_include):
            args.append(f'-I{sys_include}')
    return args


# =============================================================================
# AST Inspection Utilities
# =============================================================================

def get_template_parameters(node):
    """Extract template parameter names from a class template node."""
    template_params = []
    if node.kind != clang.cindex.CursorKind.CLASS_TEMPLATE: # type: ignore
        return template_params
    
    for child in node.get_children():
        if child.kind == clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER: # type: ignore
            param_name = child.spelling or f"T{len(template_params)}"
            template_params.append(param_name)
        elif child.kind == clang.cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER: # type: ignore
            param_name = child.spelling or f"N{len(template_params)}"
            template_params.append(param_name)
    return template_params


def get_class_metadata(node, header_path):
    """Extract @registry_name, @registry_aliases, and @category from comments above class."""
    if not node.location or not node.location.line:
        return {}
    
    try:
        with open(header_path, 'r') as f:
            lines = f.readlines()
        
        class_line = node.location.line - 1  # 0-indexed
        metadata = {}
        
        for i in range(max(0, class_line - 15), class_line):
            line = lines[i]
            if match := re.match(r'//\s*@registry_name:\s*(.+)', line):
                metadata['registry_name'] = match.group(1).strip()
            elif match := re.match(r'//\s*@registry_aliases:\s*(.+)', line):
                metadata['aliases'] = [a.strip() for a in match.group(1).split(',')]
            elif match := re.match(r'//\s*@category:\s*(.+)', line):
                metadata['category'] = match.group(1).strip()
        
        return metadata
    except Exception:
        return {}


def get_class_constructors(node, class_name):
    """
    Extract constructor information from a class node.
    Returns: list of constructors, each as [(param_type, param_name, default_value), ...]
    """
    constructors = []
    any_constructor_seen = False
    header_path = str(node.location.file) if node.location and node.location.file else None
    
    for child in node.get_children():
        if child.kind != clang.cindex.CursorKind.CONSTRUCTOR: # type: ignore
            continue

        any_constructor_seen = True

        source_access = None
        if header_path and child.location and child.location.line:
            source_access = _get_member_access_from_source(node, header_path, child.location.line)

        if source_access is not None:
            if source_access != "public":
                continue
        else:
            access = getattr(child, 'access_specifier', None)
            if access != clang.cindex.AccessSpecifier.PUBLIC: # type: ignore
                continue
        
        params = []
        for param in child.get_children():
            if param.kind != clang.cindex.CursorKind.PARM_DECL: # type: ignore
                continue
            
            param_name = param.spelling or "unnamed"
            param_type = _extract_param_type(param, param_name)
            default_value = _extract_default_value(param)
            params.append((param_type, param_name, default_value))
        
        constructors.append(params)
    
    if not constructors and not any_constructor_seen:
        constructors.append([])  # Default constructor
    
    _print_constructors(class_name, constructors)
    return constructors


def _get_member_access_from_source(class_node, header_path, member_line):
    """Best-effort source-based access resolution for a member line within a class body.
    Returns 'public'/'private'/'protected' or None if it cannot be determined.
    """
    try:
        with open(header_path, 'r') as f:
            lines = f.readlines()
    except Exception:
        return None

    if not class_node.location or not class_node.location.line:
        return None

    start_idx = max(0, class_node.location.line - 1)
    class_name = class_node.spelling

    class_decl_pattern = rf"\b(class|struct)\s+{re.escape(class_name)}\b"
    decl_idx = None
    for i in range(start_idx, min(len(lines), start_idx + 60)):
        if re.search(class_decl_pattern, lines[i]):
            decl_idx = i
            break
    if decl_idx is None:
        decl_idx = start_idx

    open_idx = None
    for i in range(decl_idx, len(lines)):
        if '{' in lines[i]:
            open_idx = i
            break
        if ';' in lines[i]:
            return None
    if open_idx is None:
        return None

    decl_text = ''.join(lines[decl_idx:open_idx + 1])
    default_access = 'public' if re.search(rf"\bstruct\s+{re.escape(class_name)}\b", decl_text) else 'private'

    depth = 0
    access = default_access

    for line_idx in range(open_idx, len(lines)):
        line = lines[line_idx]

        if depth == 1:
            m = re.match(r'\s*(public|private|protected)\s*:\s*$', line)
            if m:
                access = m.group(1)

        for ch in line:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth <= 0:
                    return None

        if depth == 1 and (line_idx + 1) == member_line:
            return access

    return None


def _extract_param_type(param_node, param_name):
    """Recover the declared parameter type, preferring source tokens over clang spelling."""
    try:
        param_tokens = [t.spelling for t in param_node.get_tokens()]
        reconstructed = _extract_param_type_from_tokens(param_tokens, param_name)
        if reconstructed:
            return reconstructed
    except Exception:
        pass

    return param_node.type.spelling if param_node.type else "unknown"


def _extract_param_type_from_tokens(param_tokens, param_name):
    """Extract the parameter type from full parameter tokens."""
    if not param_tokens or not param_name:
        return None

    cutoff = len(param_tokens)
    if '=' in param_tokens:
        cutoff = param_tokens.index('=')

    type_tokens = param_tokens[:cutoff]
    for idx in range(len(type_tokens) - 1, -1, -1):
        if type_tokens[idx] == param_name:
            type_tokens = type_tokens[:idx]
            break

    value = ' '.join(type_tokens).strip()
    if not value:
        return None

    value = re.sub(r'\s*::\s*', '::', value)
    value = re.sub(r'\s*<\s*', '<', value)
    value = re.sub(r'\s*>\s*', '>', value)
    value = re.sub(r'\s*,\s*', ', ', value)
    value = re.sub(r'\s*&\s*', '&', value)
    value = re.sub(r'\s*\*\s*', '*', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value or None


def _extract_default_value(param_node):
    """Extract default value from a parameter declaration node."""
    default_expr_kinds = (
        clang.cindex.CursorKind.INTEGER_LITERAL, # type: ignore
        clang.cindex.CursorKind.FLOATING_LITERAL, # type: ignore
        clang.cindex.CursorKind.STRING_LITERAL, # type: ignore
        clang.cindex.CursorKind.CXX_BOOL_LITERAL_EXPR, # type: ignore
        clang.cindex.CursorKind.CXX_NULL_PTR_LITERAL_EXPR, # type: ignore
        clang.cindex.CursorKind.UNEXPOSED_EXPR, # type: ignore
        clang.cindex.CursorKind.CALL_EXPR, # type: ignore
        clang.cindex.CursorKind.UNARY_OPERATOR, # type: ignore
    )
    
    for child in param_node.get_children():
        if child.kind in default_expr_kinds:
            try:
                tokens = list(child.get_tokens())
                if tokens:
                    value = _normalize_default_value_text(' '.join(t.spelling for t in tokens))
                    if value and value != "=":
                        return value
            except Exception:
                pass
            break

    # Fallback for complex/unexposed defaults: parse from full parameter tokens.
    # This handles cases where libclang exposes only "=" as an unexposed expression.
    try:
        param_tokens = [t.spelling for t in param_node.get_tokens()]
        value = _normalize_default_value_text(_extract_default_from_param_tokens(param_tokens))
        if value:
            return value
    except Exception:
        pass

    return None


def _normalize_default_value_text(value):
    """Normalize extracted default expressions to valid RHS-only code."""
    if value is None:
        return None

    value = value.strip()
    value = re.sub(r'^(?:=\s*)+', '', value)
    return value or None


def _extract_default_from_param_tokens(param_tokens):
    """Extract default expression RHS from full parameter token sequence."""
    if not param_tokens or '=' not in param_tokens:
        return None

    eq_idx = param_tokens.index('=')
    start_idx = eq_idx + 1
    while start_idx < len(param_tokens) and param_tokens[start_idx] == '=':
        start_idx += 1

    rhs_tokens = []
    paren = bracket = brace = angle = 0

    for tok in param_tokens[start_idx:]:
        # Stop at parameter separators / end-of-parameter-list at top-level.
        if tok == ',' and paren == bracket == brace == angle == 0:
            break
        if tok == ')' and paren == bracket == brace == angle == 0:
            break

        rhs_tokens.append(tok)

        if tok == '(':
            paren += 1
        elif tok == ')':
            paren = max(0, paren - 1)
        elif tok == '[':
            bracket += 1
        elif tok == ']':
            bracket = max(0, bracket - 1)
        elif tok == '{':
            brace += 1
        elif tok == '}':
            brace = max(0, brace - 1)
        elif tok == '<' and paren == 0 and bracket == 0 and brace == 0:
            angle += 1
        elif tok == '>' and paren == 0 and bracket == 0 and brace == 0:
            angle = max(0, angle - 1)

    value = _normalize_default_value_text(' '.join(rhs_tokens))
    if value and value != "=":
        return value
    return None


def _print_constructors(class_name, constructors):
    """Print constructor information for debugging."""
    if constructors and constructors != [[]]:
        print(f"    Found {len(constructors)} constructor(s) for {class_name}:")
        for i, params in enumerate(constructors):
            param_strs = []
            for ptype, pname, pdefault in params:
                if pdefault:
                    param_strs.append(f"{ptype} {pname} = {pdefault}")
                else:
                    param_strs.append(f"{ptype} {pname}")
            print(f"      Constructor {i+1}: {class_name}({', '.join(param_strs)})")
    else:
        print(f"    No explicit constructors found for {class_name} (default constructor available)")


# =============================================================================
# Inheritance Detection
# =============================================================================

def is_subclass_of(node, base_name, seen=None):
    """Check if node inherits from base_name using AST, following intermediate bases."""
    if seen is None:
        seen = set()

    node_id = node.hash if hasattr(node, 'hash') else id(node)
    if node_id in seen:
        return False
    seen.add(node_id)

    for child in node.get_children():
        if child.kind != clang.cindex.CursorKind.CXX_BASE_SPECIFIER: # type: ignore
            continue
        
        candidates = []
        try:
            decl = child.type.get_declaration()
            if decl and decl.spelling:
                candidates.append(decl.spelling)
        except Exception:
            pass
        
        if tsp := getattr(child.type, 'spelling', None):
            candidates.append(tsp)
        if disp := getattr(child, 'displayname', None):
            candidates.append(disp)
        
        for name in candidates:
            if not name:
                continue
            name = re.sub(r'<.*>$', '', name).strip()
            simple = name.split('::')[-1]
            if simple == base_name or name == base_name:
                return True

        try:
            decl = child.type.get_declaration()
            if decl and decl.spelling and is_subclass_of(decl, base_name, seen):
                return True
        except Exception:
            pass
    return False


def text_inherits(header_path, node_name, base_name):
    """Fallback: check inheritance via regex in source text."""
    try:
        with open(header_path, 'r') as f:
            txt = f.read()
    except Exception:
        return False
    
    pattern = rf"class\s+{re.escape(node_name)}[^{{]*:\s*[^{{;]*\b{re.escape(base_name)}\b"
    return re.search(pattern, txt, flags=re.MULTILINE | re.DOTALL) is not None


def class_body_contains_pure_virtual(node, header_path):
    """Return True if the class body text contains an explicit pure-virtual declaration."""
    if not node.location or not node.location.line:
        return False

    try:
        with open(header_path, 'r') as f:
            lines = f.readlines()
    except Exception:
        return False

    start_line = max(0, node.location.line - 1)
    text = ''.join(lines[start_line:])
    class_match = re.search(
        rf"\b(class|struct)\s+{re.escape(node.spelling)}\b.*?\{{",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not class_match:
        return False

    body_start = class_match.end()
    depth = 1
    idx = body_start
    while idx < len(text) and depth > 0:
        if text[idx] == '{':
            depth += 1
        elif text[idx] == '}':
            depth -= 1
        idx += 1

    body = text[body_start:idx - 1] if depth == 0 else text[body_start:]
    pure_virtual_pattern = (
        r"\bvirtual\b[\s\S]*?\)\s*"
        r"(?:const\s*)?"
        r"(?:noexcept(?:\s*\([^)]*\))?\s*)?"
        r"(?:override\s*)?"
        r"(?:final\s*)?"
        r"=\s*0\s*;"
    )
    return re.search(pure_virtual_pattern, body, flags=re.MULTILINE) is not None


def is_effectively_abstract(node, header_path):
    """Best-effort abstract-class detection, including template classes."""
    try:
        if node.is_abstract_record():
            return True
    except Exception:
        pass

    return class_body_contains_pure_virtual(node, header_path)


def is_base_class_declaration(node, class_name, current_file):
    """
    Check if node declares the base class in current file.
    Returns: (is_base_class, template_params)
    """
    if node.kind not in (
        clang.cindex.CursorKind.CLASS_DECL, # type: ignore
        clang.cindex.CursorKind.STRUCT_DECL, # type: ignore
        clang.cindex.CursorKind.CLASS_TEMPLATE, # type: ignore
    ):
        return False, []
    
    name = re.sub(r'<.*>$', '', node.spelling or "").strip()
    simple = name.split('::')[-1]
    
    if simple != class_name and name != class_name:
        return False, []
    
    node_file = str(node.location.file) if node.location.file else ""
    if not (node_file.endswith(current_file) or current_file.endswith(node_file)):
        return False, []
    
    print(f"  {node.spelling} (defined in {node_file})")
    template_params = get_template_parameters(node)
    if template_params:
        print(f"    Template parameters: {template_params}")
    
    return True, template_params


# =============================================================================
# Path Utilities
# =============================================================================

def normalize_include_path(path):
    """Convert absolute or relative path to relative include path."""
    if "include/" in path:
        return path[path.rfind("include/") + len("include/"):]
    if path.startswith("include/"):
        return path[len("include/"):]
    return path


def extract_base_class_from_type(param_type, base_classes):
    """
    Extract base class name from a parameter type if it matches any known base class.
    Handles pointers, references, templates, and qualified names.
    
    Example: "MLCouplingNormalization<In, Out>*" -> "MLCouplingNormalization"
    """
    # Remove pointer/reference/const/volatile qualifiers
    clean_type = re.sub(r'\s*(const|volatile)\s+', '', param_type)
    clean_type = re.sub(r'[*&]+\s*$', '', clean_type).strip()
    
    # Extract base type name (before template arguments)
    match = re.match(r'([^<]+)', clean_type)
    if match:
        base_type = match.group(1).strip()
        # Remove namespace qualifiers and get simple name
        simple_name = base_type.split('::')[-1]
        
        # Check if it matches any base class
        for base_class in base_classes:
            if simple_name == base_class or base_type == base_class:
                return base_class
    
    return None


def _normalize_type_for_display(typ: str) -> str:
    """Normalize type spacing for human-readable output.
    Removes spaces before/after pointer stars so `In *` -> `In*`,
    `MLCouplingData<In> *` -> `MLCouplingData<In>*`, etc.
    """
    if not typ:
        return typ
    # Remove spaces around '*'
    typ = re.sub(r"\s*\*\s*", "*", typ)
    # Collapse multiple spaces
    typ = re.sub(r"\s+", " ", typ).strip()
    return typ


def _strip_cvref(param_type: str) -> str:
    """Strip top-level const/volatile and reference qualifiers from a type string."""
    clean_type = re.sub(r'\s*(const|volatile)\s+', '', param_type)
    clean_type = re.sub(r'([*&]+)\s*$', '', clean_type).strip()
    return re.sub(r"\s+", " ", clean_type).strip()


# =============================================================================
# Code Generation
# =============================================================================

def write_includes(f, base_classes, base_classes_found, found_classes):
    """Generate #include statements."""
    f.write("#pragma once\n\n")
    f.write("#include <string>\n#include <vector>\n#include <unordered_map>\n#include <iostream>\n#include <typeinfo>\n#include <type_traits>\n#include <limits>\n#include <cmath>\n#include <algorithm>\n#include <cctype>\n\n")
    
    # Base class includes
    for base_class in base_classes:
        if base_class in base_classes_found:
            path = normalize_include_path(base_classes_found[base_class])
            f.write(f'#include "{path}" // {base_class} \n')
    
    # Subclass includes
    for base_class in base_classes:
        f.write(f'\n// Includes for subclasses of {base_class}\n')
        for entry in found_classes[base_class]:
            cls, h = entry[0], entry[1]
            path = normalize_include_path(h)
            f.write(f'#include "{path}" // {cls} \n')
    
    f.write("\n\n\n")

def write_combined_lookup_function(f, categories):
    
    f.write("inline std::string resolve_class_name(const std::string& name_or_alias) {\n")
    f.write("    // This function checks all categories for a matching name or alias and returns the resolved class name.\n")
    f.write(f"    std::string resolved;\n")
    
    for category in categories:
        f.write(f"    resolved = resolve_{category}_class_name(name_or_alias);\n")
        f.write(f"    if (resolved != name_or_alias) {{\n")
        f.write(f"        return resolved;\n")
        f.write(f"    }}\n")

    f.write("    return name_or_alias; // Return as-is if no mapping found in any category\n")
    f.write("}\n\n")
    
    
def write_lookup_functions(f, base_classes, base_class_categories, found_classes, class_metadata):
    """Generate name/alias lookup functions."""
    for base_class in base_classes:
        category = base_class_categories.get(base_class, base_class.lower())
        
        f.write(f"// Lookup function for {base_class} ({category})\n")
        f.write(f"// Maps registry names and aliases to actual class names\n")
        f.write(f"inline std::string resolve_{category}_class_name(const std::string& name_or_alias) {{\n")
        f.write(f"    static const std::unordered_map<std::string, std::string> lookup = {{\n")
        
        for entry in found_classes[base_class]:
            cls = entry[0]
            if cls in base_classes:
                continue
            
            metadata = class_metadata[base_class].get(cls, {})
            if 'registry_name' in metadata:
                f.write(f'        {{"{metadata["registry_name"]}", "{cls}"}},\n')
            for alias in metadata.get('aliases', []):
                f.write(f'        {{"{alias}", "{cls}"}},\n')
        
        f.write("    };\n\n")
        f.write("    auto it = lookup.find(name_or_alias);\n")
        f.write("    if (it != lookup.end()) {\n")
        f.write("        return it->second;\n")
        f.write("    }\n")
        f.write("    return name_or_alias; // Return as-is if no mapping found\n")
        f.write("}\n\n")


def write_category_lookup(f, base_classes, base_class_categories):
    """Generate category to base class lookup function."""
    f.write("// Lookup function to resolve category names to base class names\n")
    f.write("inline std::string resolve_category_to_base_class(const std::string& category) {\n")
    f.write("    static const std::unordered_map<std::string, std::string> lookup = {\n")
    
    for base_class in base_classes:
        category = base_class_categories.get(base_class, base_class.lower())
        f.write(f'        {{"{category}", "{base_class}"}},\n')
    
    f.write("    };\n\n")
    f.write("    auto it = lookup.find(category);\n")
    f.write("    if (it != lookup.end()) {\n")
    f.write("        return it->second;\n")
    f.write("    }\n")
    f.write("    return category; // Return as-is if no mapping found\n")
    f.write("}\n\n")


def write_constructor_dependencies(f, base_classes, found_classes, subclass_constructors):
    """
    Generate function that returns constructor parameter dependencies.
    For each subclass, returns list of (base_class_type, param_name) pairs.
    """
    f.write("// Get constructor parameter dependencies for a given class\n")
    f.write("// Returns pairs of (base_class_type, parameter_name) for parameters that are base classes\n")
    f.write("inline std::vector<std::pair<std::string, std::string>> get_constructor_dependencies(const std::string& class_name) {\n")
    f.write("    std::vector<std::pair<std::string, std::string>> dependencies;\n\n")
    
    first_class = True
    for base_class in base_classes:
        for entry in found_classes[base_class]:
            cls = entry[0]
            if cls in base_classes:
                continue
            
            if cls not in subclass_constructors[base_class]:
                continue
            
            # Generate if-else chain
            if first_class:
                f.write(f'    if (class_name == "{cls}") {{\n')
                first_class = False
            else:
                f.write(f'    }} else if (class_name == "{cls}") {{\n')
            
            # Process all constructors (typically just one, but handle multiple)
            constructors = subclass_constructors[base_class][cls]
            if constructors and constructors != [[]]:
                # Use the first constructor with parameters (or first constructor if all are empty)
                selected_ctor = None
                for ctor in constructors:
                    if ctor:  # Non-empty constructor
                        selected_ctor = ctor
                        break
                if not selected_ctor and constructors:
                    selected_ctor = constructors[0]
                
                if selected_ctor:
                    for param_type, param_name, _ in selected_ctor:
                        # Check if this parameter type is one of our base classes
                        matching_base = extract_base_class_from_type(param_type, base_classes)
                        if matching_base:
                            f.write(f'        dependencies.push_back({{"{matching_base}", "{param_name}"}});\n')
    
    if not first_class:
        f.write("    }\n\n")
    
    f.write("    return dependencies;\n")
    f.write("}\n\n")


def write_constructor_signatures(f, base_classes, found_classes, subclass_constructors):
    """
    Generate function that returns human-friendly constructor signatures for a class.
    This is used to print help when provided arguments do not match any constructor.
    """
    f.write("// Get constructor signatures for a given class (for help messages)\n")
    f.write("inline std::vector<std::string> get_constructor_signatures(const std::string& class_name) {\n")
    f.write("    std::vector<std::string> signatures;\n\n")

    for base_class in base_classes:
        for entry in found_classes[base_class]:
            cls = entry[0]
            if cls in base_classes:
                continue

            if cls not in subclass_constructors[base_class]:
                continue

            f.write(f'    if (class_name == "{cls}") {{\n')
            ctors = subclass_constructors[base_class][cls]
            if ctors and ctors != [[]]:
                for ctor in ctors:
                    parts = []
                    for ptype, pname, pdefault in ctor:
                        display_type = _normalize_type_for_display(ptype)
                        if pdefault:
                            parts.append(f'{display_type} {pname} = {pdefault}')
                        else:
                            parts.append(f'{display_type} {pname}')
                    sig = f"{cls}({', '.join(parts)})"
                    # Escape backslashes and double-quotes conservatively
                    safe_sig = sig.replace('\\', '\\\\').replace('"', '\\"')
                    f.write(f'        signatures.push_back("{safe_sig}");\n')
            else:
                f.write(f'        signatures.push_back("{cls}()\");\n')

            f.write("        return signatures;\n")
            f.write("    }\n\n")

    f.write("    return signatures;\n")
    f.write("}\n\n")


def write_print_constructor_help(f):
    """Generate a small helper that prints constructor signatures to stdout."""
    f.write("// Print constructor help to console/log\n")
    f.write("inline void print_constructor_help(const std::string& class_name) {\n")
    f.write("    auto sigs = get_constructor_signatures(class_name);\n")
    f.write("    if (sigs.empty()) { std::cout << \"No constructors found for \" << class_name << std::endl; return; }\n")
    f.write("    std::cout << \"Available constructors for \" << class_name << \":\" << std::endl;\n")
    f.write("    for (const auto &s : sigs) std::cout << \"  \" << s << std::endl;\n")
    f.write("}\n\n")


def write_class_hierarchy_functions(f, base_classes, found_classes):
    """
    Generate functions to query class hierarchy:
    - get_subclasses(base_class_name): returns all subclasses of a base class
    - get_superclasses(class_name): returns all superclasses up the hierarchy
    """
    # Map: subclass -> base class (for quick lookup)
    subclass_to_base = {}
    for base_class in base_classes:
        for entry in found_classes[base_class]:
            cls = entry[0]
            if cls not in base_classes:
                subclass_to_base[cls] = base_class
    
    # Write get_subclasses function
    f.write("// Get all subclasses of a given base class name\n")
    f.write("inline std::vector<std::string> get_subclasses(const std::string& base_class_name) {\n")
    f.write("    std::vector<std::string> subclasses;\n\n")
    
    for base_class in base_classes:
        f.write(f'    if (base_class_name == "{base_class}") {{\n')
        for entry in found_classes[base_class]:
            cls = entry[0]
            if cls not in base_classes:
                f.write(f'        subclasses.push_back("{cls}");\n')
        f.write("    }\n\n")
    
    f.write("    return subclasses;\n")
    f.write("}\n\n")
    
    # Write get_superclasses function
    f.write("// Get all superclasses of a given class name (from subclass up to base class)\n")
    f.write("inline std::vector<std::string> get_superclasses(const std::string& class_name) {\n")
    f.write("    std::vector<std::string> superclasses;\n")
    f.write("    static const std::unordered_map<std::string, std::string> hierarchy = {\n")
    
    for subclass, base_class in subclass_to_base.items():
        f.write(f'        {{"{subclass}", "{base_class}"}},\n')
    
    f.write("    };\n\n")
    f.write("    auto it = hierarchy.find(class_name);\n")
    f.write("    if (it != hierarchy.end()) {\n")
    f.write("        std::string current = it->second;\n")
    f.write("        superclasses.push_back(current);\n")
    f.write("        // Note: Currently only supports single inheritance (one level up).\n")
    f.write("        // If multi-level hierarchies are needed, extend this recursively.\n")
    f.write("    }\n")
    f.write("    return superclasses;\n")
    f.write("}\n\n")


def _get_template_strings(base_class, template_params):
    """Get template declaration and argument strings."""
    if not template_params:
        return f"{base_class}*", ""
    
    template_decl = ", ".join(f"typename {p}" for p in template_params)
    template_args = ", ".join(template_params)
    return f"template<{template_decl}>\n{base_class}<{template_args}>*", template_args


def _get_class_instantiation(cls, subclass_template_params, base_template_params):
    """Get instantiation string for a class (with template args if needed)."""
    if subclass_template_params:
        args = ", ".join(base_template_params[:len(subclass_template_params)])
        return f"{cls}<{args}>"
    return cls


def write_type_identification_functions(f, base_classes, template_parameters, found_classes):
    """
    Generate typeid-based runtime type identification functions.
    For each base class, emits:
      get_type_name(const Base* obj)   -> std::string
      get_type_name(const Base& obj)   -> std::string  (delegates to pointer overload)
    """
    f.write("// ---------------------------------------------------------------------------\n")
    f.write("// Runtime type identification via typeid comparison\n")
    f.write("// Returns the human-readable class name for a given (possibly polymorphic) object.\n")
    f.write("// ---------------------------------------------------------------------------\n\n")

    for base_class in base_classes:
        base_template_params = template_parameters.get(base_class, [])

        if base_template_params:
            template_decl = ", ".join(f"typename {p}" for p in base_template_params)
            template_args  = ", ".join(base_template_params)
            ptr_prefix = f"template<{template_decl}>\n"
            base_ptr  = f"const {base_class}<{template_args}>*"
            base_ref  = f"const {base_class}<{template_args}>&"
        else:
            template_decl = ""
            template_args  = ""
            ptr_prefix = ""
            base_ptr  = f"const {base_class}*"
            base_ref  = f"const {base_class}&"

        # --- pointer overload ---
        f.write(f"{ptr_prefix}inline std::string get_type_name({base_ptr} obj) {{\n")
        f.write("    if (!obj) return \"nullptr\";\n")

        for entry in found_classes[base_class]:
            cls, _, subclass_tparams = entry
            if cls in base_classes:
                continue
            if subclass_tparams and base_template_params:
                args = ", ".join(base_template_params[:len(subclass_tparams)])
                cls_type = f"{cls}<{args}>"
            else:
                cls_type = cls
            f.write(f'    if (typeid(*obj) == typeid({cls_type})) return "{cls}";\n')

        # Fallback: the base class itself (concrete or unknown derived)
        if base_template_params:
            base_type = f"{base_class}<{template_args}>"
        else:
            base_type = base_class
        f.write(f'    if (typeid(*obj) == typeid({base_type})) return "{base_class}";\n')
        f.write('    return "unknown";\n')
        f.write("}\n\n")

        # --- reference overload (delegates) ---
        f.write(f"{ptr_prefix}inline std::string get_type_name({base_ref} obj) {{\n")
        f.write("    return get_type_name(&obj);\n")
        f.write("}\n\n")


def write_config_param_cast_helper(f):
    """Generate a template helper that casts a typed void* to a target type at runtime."""
    f.write("enum class ConfigCastMode : int { Strict, Relaxed };\n\n")
    f.write("inline ConfigCastMode& config_cast_mode_storage() {\n")
    f.write("    static ConfigCastMode mode = ConfigCastMode::Relaxed;\n")
    f.write("    return mode;\n")
    f.write("}\n\n")
    f.write("inline void set_config_cast_mode(ConfigCastMode mode) {\n")
    f.write("    config_cast_mode_storage() = mode;\n")
    f.write("}\n\n")
    f.write("inline ConfigCastMode get_config_cast_mode() {\n")
    f.write("    return config_cast_mode_storage();\n")
    f.write("}\n\n")
    f.write("inline std::string config_cast_to_lower(std::string value) {\n")
    f.write("    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });\n")
    f.write("    return value;\n")
    f.write("}\n\n")
    f.write("inline bool config_try_parse_bool(const std::string& value, bool& out) {\n")
    f.write("    auto lower = config_cast_to_lower(value);\n")
    f.write("    if (lower == \"true\" || lower == \"1\") { out = true; return true; }\n")
    f.write("    if (lower == \"false\" || lower == \"0\") { out = false; return true; }\n")
    f.write("    return false;\n")
    f.write("}\n\n")
    f.write("template<typename T>\n")
    f.write("inline T config_cast_from_int64(int64_t value, ConfigCastMode mode) {\n")
    f.write("    if constexpr (std::is_same_v<T, bool>) {\n")
    f.write("        if (mode == ConfigCastMode::Strict && value != 0 && value != 1) {\n")
    f.write("            throw std::runtime_error(\"Strict cast failed: int64_t to bool requires 0 or 1.\");\n")
    f.write("        }\n")
    f.write("        return value != 0;\n")
    f.write("    } else if constexpr (std::is_integral_v<T>) {\n")
    f.write("        if (value < static_cast<int64_t>(std::numeric_limits<T>::min()) || value > static_cast<int64_t>(std::numeric_limits<T>::max())) {\n")
    f.write("            throw std::runtime_error(\"Cast failed: int64_t value out of range for target integral type.\");\n")
    f.write("        }\n")
    f.write("        return static_cast<T>(value);\n")
    f.write("    } else if constexpr (std::is_floating_point_v<T>) {\n")
    f.write("        return static_cast<T>(value);\n")
    f.write("    } else if constexpr (std::is_same_v<T, std::string>) {\n")
    f.write("        if (mode == ConfigCastMode::Strict) {\n")
    f.write("            throw std::runtime_error(\"Strict cast failed: int64_t to string is not allowed.\");\n")
    f.write("        }\n")
    f.write("        return std::to_string(value);\n")
    f.write("    }\n")
    f.write("    throw std::runtime_error(\"Unsupported target type for int64_t config cast.\");\n")
    f.write("}\n\n")
    f.write("template<typename T>\n")
    f.write("inline T config_cast_from_double(double value, ConfigCastMode mode) {\n")
    f.write("    if constexpr (std::is_same_v<T, bool>) {\n")
    f.write("        if (mode == ConfigCastMode::Strict && value != 0.0 && value != 1.0) {\n")
    f.write("            throw std::runtime_error(\"Strict cast failed: double to bool requires 0.0 or 1.0.\");\n")
    f.write("        }\n")
    f.write("        return value != 0.0;\n")
    f.write("    } else if constexpr (std::is_integral_v<T>) {\n")
    f.write("        if (mode == ConfigCastMode::Strict && std::floor(value) != value) {\n")
    f.write("            throw std::runtime_error(\"Strict cast failed: double to integral requires an integer-valued source.\");\n")
    f.write("        }\n")
    f.write("        if (value < static_cast<double>(std::numeric_limits<T>::min()) || value > static_cast<double>(std::numeric_limits<T>::max())) {\n")
    f.write("            throw std::runtime_error(\"Cast failed: double value out of range for target integral type.\");\n")
    f.write("        }\n")
    f.write("        return static_cast<T>(value);\n")
    f.write("    } else if constexpr (std::is_floating_point_v<T>) {\n")
    f.write("        if constexpr (std::is_same_v<T, float>) {\n")
    f.write("            if (mode == ConfigCastMode::Strict && (value < -std::numeric_limits<float>::max() || value > std::numeric_limits<float>::max())) {\n")
    f.write("                throw std::runtime_error(\"Strict cast failed: double out of range for float.\");\n")
    f.write("            }\n")
    f.write("        }\n")
    f.write("        return static_cast<T>(value);\n")
    f.write("    } else if constexpr (std::is_same_v<T, std::string>) {\n")
    f.write("        if (mode == ConfigCastMode::Strict) {\n")
    f.write("            throw std::runtime_error(\"Strict cast failed: double to string is not allowed.\");\n")
    f.write("        }\n")
    f.write("        return std::to_string(value);\n")
    f.write("    }\n")
    f.write("    throw std::runtime_error(\"Unsupported target type for double config cast.\");\n")
    f.write("}\n\n")
    f.write("template<typename T>\n")
    f.write("inline T config_cast_from_bool(bool value, ConfigCastMode mode) {\n")
    f.write("    if constexpr (std::is_same_v<T, bool>) {\n")
    f.write("        return value;\n")
    f.write("    } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {\n")
    f.write("        if (mode == ConfigCastMode::Strict) {\n")
    f.write("            throw std::runtime_error(\"Strict cast failed: bool to numeric is not allowed.\");\n")
    f.write("        }\n")
    f.write("        return static_cast<T>(value ? 1 : 0);\n")
    f.write("    } else if constexpr (std::is_same_v<T, std::string>) {\n")
    f.write("        if (mode == ConfigCastMode::Strict) {\n")
    f.write("            throw std::runtime_error(\"Strict cast failed: bool to string is not allowed.\");\n")
    f.write("        }\n")
    f.write("        return value ? std::string(\"true\") : std::string(\"false\");\n")
    f.write("    }\n")
    f.write("    throw std::runtime_error(\"Unsupported target type for bool config cast.\");\n")
    f.write("}\n\n")
    f.write("template<typename T>\n")
    f.write("inline T config_cast_from_string(const std::string& value, ConfigCastMode mode) {\n")
    f.write("    if constexpr (std::is_same_v<T, std::string>) {\n")
    f.write("        return value;\n")
    f.write("    }\n")
    f.write("\n")
    f.write("    if (mode == ConfigCastMode::Strict) {\n")
    f.write("        throw std::runtime_error(\"Strict cast failed: string source only supports std::string target.\");\n")
    f.write("    }\n")
    f.write("\n")
    f.write("    if constexpr (std::is_same_v<T, bool>) {\n")
    f.write("        bool parsed = false;\n")
    f.write("        if (!config_try_parse_bool(value, parsed)) {\n")
    f.write("            throw std::runtime_error(\"Relaxed cast failed: could not parse string as bool: \" + value);\n")
    f.write("        }\n")
    f.write("        return parsed;\n")
    f.write("    }\n")
    f.write("\n")
    f.write("    if constexpr (std::is_integral_v<T>) {\n")
    f.write("        size_t pos = 0;\n")
    f.write("        long long parsed = std::stoll(value, &pos);\n")
    f.write("        if (pos != value.size()) {\n")
    f.write("            throw std::runtime_error(\"Relaxed cast failed: trailing characters in integral string: \" + value);\n")
    f.write("        }\n")
    f.write("        if (parsed < static_cast<long long>(std::numeric_limits<T>::min()) || parsed > static_cast<long long>(std::numeric_limits<T>::max())) {\n")
    f.write("            throw std::runtime_error(\"Relaxed cast failed: parsed integer out of target range.\");\n")
    f.write("        }\n")
    f.write("        return static_cast<T>(parsed);\n")
    f.write("    }\n")
    f.write("\n")
    f.write("    if constexpr (std::is_floating_point_v<T>) {\n")
    f.write("        size_t pos = 0;\n")
    f.write("        double parsed = std::stod(value, &pos);\n")
    f.write("        if (pos != value.size()) {\n")
    f.write("            throw std::runtime_error(\"Relaxed cast failed: trailing characters in floating string: \" + value);\n")
    f.write("        }\n")
    f.write("        return static_cast<T>(parsed);\n")
    f.write("    }\n")
    f.write("\n")
    f.write("    throw std::runtime_error(\"Unsupported target type for string config cast.\");\n")
    f.write("}\n\n")
    f.write("// Helper to extract and cast a config parameter based on its runtime type tag.\n")
    f.write("// Type tags: 0 = no static cast, 1 = int64_t, 2 = double, 3 = std::string (char*), 4 = bool\n")
    f.write("template<typename T>\n")
    f.write("T config_param_cast(const std::pair<int, void*>& param) {\n")
    f.write("    const auto mode = get_config_cast_mode();\n")
    f.write("    switch (param.first) {\n")
    f.write("        case 0:\n")
    f.write("            return *reinterpret_cast<T*>(param.second); // No static cast, just reinterpret\n")
    f.write("        case 1:\n")
    f.write("            return config_cast_from_int64<T>(*reinterpret_cast<int64_t*>(param.second), mode);\n")
    f.write("        case 2:\n")
    f.write("            return config_cast_from_double<T>(*reinterpret_cast<double*>(param.second), mode);\n")
    f.write("        case 3:\n")
    f.write("            return config_cast_from_string<T>(std::string(reinterpret_cast<char*>(param.second)), mode);\n")
    f.write("        case 4:\n")
    f.write("            return config_cast_from_bool<T>(*reinterpret_cast<bool*>(param.second), mode);\n")
    f.write("        default:\n")
    f.write("            throw std::runtime_error(\"Unsupported type tag for config cast: \" + std::to_string(param.first));\n")
    f.write("    }\n")
    f.write("}\n\n")


def write_factory_functions(f, base_classes, template_parameters, base_class_categories,
                            found_classes, subclass_constructors):
    """Generate factory functions for creating instances."""
    for base_class in base_classes:
        base_template_params = template_parameters.get(base_class, ['In', 'Out'])
        template_str, template_args = _get_template_strings(base_class, base_template_params)
        category = base_class_categories.get(base_class, base_class.lower())
        
        # Filter valid entries
        entries = [(cls, h, tparams) for cls, h, tparams in found_classes[base_class]
                   if cls not in base_classes and cls in subclass_constructors[base_class]]
        
        # Map-based factory with name resolution
        _write_map_factory(f, base_class, template_str, template_args, category,
                           base_template_params, entries, subclass_constructors)


def _is_mlcoupling_data_type(param_type):
    """Check if parameter type is MLCouplingData<...>."""
    clean_type = re.sub(r'\s*(const|volatile)\s+', '', param_type)
    clean_type = re.sub(r'[*&]+\s*$', '', clean_type).strip()
    return clean_type.startswith('MLCouplingData<')


def _is_pointer_to_known_class(param_type):
    """Check if parameter type is a pointer to a known class (not a primitive)."""
    clean_type = param_type.strip()
    if not clean_type.endswith('*'):
        return False
    # Remove the pointer and check if it's a class-like type (starts with uppercase or MLCoupling)
    base_type = re.sub(r'[*&]+\s*$', '', clean_type).strip()
    base_type = re.sub(r'<.*>$', '', base_type).strip()
    # Primitive types and their variants are not "known classes"
    primitives = {'int', 'float', 'double', 'bool', 'char', 'int64_t', 'int32_t',
                  'uint64_t', 'uint32_t', 'size_t', 'std::string'}
    return base_type not in primitives and not base_type.startswith('std::')


def _is_reference_type(param_type):
    """Check if parameter type is an lvalue/rvalue reference."""
    return param_type.strip().endswith('&')


def _can_use_config_param_cast(param_type):
    """Return True if parameter type is supported by config_param_cast<T>."""
    clean_type = _strip_cvref(param_type)

    supported = {
        'bool', 'char',
        'short', 'unsigned short',
        'int', 'unsigned int',
        'long', 'unsigned long',
        'long long', 'unsigned long long',
        'int32_t', 'uint32_t', 'int64_t', 'uint64_t', 'size_t',
        'float', 'double',
        'std::string', 'string',
    }
    return clean_type in supported

def _write_map_factory(f, base_class, template_str, template_args, category,
                       base_template_params, entries, subclass_constructors):
    """Generate unordered_map<string, pair<int,void*>> parameter factory with name resolution."""
    f.write(f"{template_str} create_instance_{base_class.lower()}(const std::string &class_name, const std::unordered_map<std::string, std::pair<int, void*>>& parameter) {{\n")
    f.write(f"    // Resolve name or alias to actual class name\n")
    f.write(f"    std::string resolved_class_name = resolve_{category}_class_name(class_name);\n\n")
    
    for i, (cls, _, subclass_template_params) in enumerate(entries):
        if i == 0:
            f.write(f'    if (resolved_class_name == "{cls}") {{\n')
        else:
            f.write(f'    }} else if (resolved_class_name == "{cls}") {{\n')
        
        cls_inst = _get_class_instantiation(cls, subclass_template_params, base_template_params)
        
        for ctor in subclass_constructors[base_class].get(cls, []):
            param_count = len(ctor)
            params_with_defaults = sum(1 for _, _, d in ctor if d is not None)
            params_without_defaults = param_count - params_with_defaults
            
            param_docs = [f"{_normalize_type_for_display(t)} {n} = {d}" if d else f"{_normalize_type_for_display(t)} {n}" for t, n, d in ctor]
            
            f.write(f'        // Constructor with {param_count} parameter(s)\n')
            f.write(f'        // Parameters: {", ".join(param_docs)}\n')
            
            if params_with_defaults > 0:
                f.write(f'        if (parameter.size() >= {params_without_defaults} && parameter.size() <= {param_count}) {{\n')
            else:
                f.write(f'        if (parameter.size() == {param_count}) {{\n')
            
            f.write(f'            try {{\n')
            
            # Generate parameter extraction code using config_param_cast
            param_args = []
            debug_outputs = []
            for ptype, pname, pdefault in ctor:
                storage_type = _strip_cvref(ptype)
                if _is_mlcoupling_data_type(ptype):
                    # For MLCouplingData, cast directly from void* (composite object, no type tag dispatch)
                    param_args.append(f'*reinterpret_cast<{storage_type}*>(parameter.at("{pname}").second)')
                    debug_outputs.append(f'"{pname}=" << (*reinterpret_cast<{storage_type}*>(parameter.at("{pname}").second))')
                elif _is_pointer_to_known_class(ptype):
                    # For pointers to known classes (e.g. MLCouplingNormalization<In,Out>*),
                    # the config already stores the raw object pointer in parameter.second.
                    param_args.append(f'reinterpret_cast<{ptype}>(parameter.at("{pname}").second)')
                    debug_outputs.append(f'"{pname}=" << reinterpret_cast<{ptype}>(parameter.at("{pname}").second)')
                elif _can_use_config_param_cast(ptype):
                    cast_type = storage_type
                    if pdefault:
                        # Parameter with default: check if present, use config_param_cast or default
                        param_args.append(f'parameter.find("{pname}") != parameter.end() ? config_param_cast<{cast_type}>(parameter.at("{pname}")) : ({cast_type}){pdefault}')
                        debug_outputs.append(f'"{pname}=" << (parameter.find("{pname}") != parameter.end() ? config_param_cast<{cast_type}>(parameter.at("{pname}")) : ({cast_type}){pdefault})')
                    else:
                        # Required parameter: use config_param_cast
                        param_args.append(f'config_param_cast<{cast_type}>(parameter.at("{pname}"))')
                        debug_outputs.append(f'"{pname}=" << config_param_cast<{cast_type}>(parameter.at("{pname}"))')
                elif pdefault:
                    # Opaque/non-primitive parameter with default: pass via typed pointer, fallback to default expression
                    param_args.append(f'parameter.find("{pname}") != parameter.end() ? *reinterpret_cast<{storage_type}*>(parameter.at("{pname}").second) : ({storage_type}){pdefault}')
                    debug_outputs.append(f'"{pname}=<" << (parameter.find("{pname}") != parameter.end() ? "provided" : "default") << ">"')
                else:
                    # Opaque/non-primitive required parameter: pass via typed pointer
                    param_args.append(f'*reinterpret_cast<{storage_type}*>(parameter.at("{pname}").second)')
                    debug_outputs.append(f'"{pname}=<provided>"')
            
            params_str = ', '.join(param_args)
            
            # Build debug output statement with proper << chaining
            debug_line = 'std::cout << "Creating instance of ' + cls + ' with parameters: "'
            for idx, debug_out in enumerate(debug_outputs):
                debug_line += " << " + ("\", \"" if idx > 0 else "") + debug_out
            debug_line += ' << std::endl;'
            
            f.write(f'                {debug_line}\n')
            f.write(f'                return new {cls_inst}({params_str});\n')
            f.write(f'            }} catch (...) {{\n')
            f.write(f'                // Handle exceptions if necessary\n')
            f.write(f'            }}\n')
            f.write(f'        }}\n')
        
        f.write(f'        return nullptr;\n')
    
    if entries:
        f.write("    }\n")
    
    f.write("    return nullptr;\n}\n\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def generate():
    if len(sys.argv) < 4:
        print("Usage: generate_registry.py <output_path> <base_classes> <header_files...>")
        sys.exit(1)
    
    output_path = sys.argv[1]
    base_class = sys.argv[2]
    headers = sys.argv[3:]
    
    base_classes = base_class.split(",") if "," in base_class else [base_class]

    # Initialize libclang
    index = init_libclang()
    parse_args = get_parse_args()
    
    # Data structures
    found_classes = {bc: [] for bc in base_classes}
    base_classes_found = {}
    subclass_constructors = {bc: {} for bc in base_classes}
    template_parameters = {bc: [] for bc in base_classes}
    class_metadata = {bc: {} for bc in base_classes}
    base_class_categories = {}

    print("Header files to scan:", headers)
    
    for h in headers:
        print("Parsing:", h)
        tu = index.parse(h, args=parse_args)
        for node in tu.cursor.walk_preorder():
            if node.kind not in (clang.cindex.CursorKind.CLASS_DECL, # type: ignore
                                 clang.cindex.CursorKind.STRUCT_DECL, # type: ignore
                                 clang.cindex.CursorKind.CLASS_TEMPLATE): # type: ignore
                continue
            
            for base_class in base_classes:
                is_base, base_template_params = is_base_class_declaration(node, base_class, h)
                if is_base:
                    print(f"File {h} contains class {base_class}")
                    base_classes_found[base_class] = h
                    template_parameters[base_class] = base_template_params
                    
                    # Extract category metadata for base class
                    base_metadata = get_class_metadata(node, h)
                    if 'category' in base_metadata:
                        base_class_categories[base_class] = base_metadata['category']
                        print(f"    Base class category: {base_metadata['category']}")
                    
                    continue
                
                is_sub = is_subclass_of(node, base_class)
                # Fallback to text-based check if AST didn't expose base specifier
                if not is_sub and node.spelling:
                    is_sub = text_inherits(h, node.spelling, base_class)

                if is_sub and node.spelling and not is_effectively_abstract(node, h):
                    if node.spelling != base_class:
                        # Only process classes defined in the current header file (not included files)
                        node_file = str(node.location.file) if node.location.file else ""
                        if not (node_file.endswith(h) or h.endswith(node_file)):
                            continue
                        
                        # Get template parameters for this subclass
                        subclass_template_params = get_template_parameters(node)
                        if subclass_template_params:
                            print(f"    Subclass {node.spelling} template parameters: {subclass_template_params}")
                        
                        # Get metadata for this subclass
                        metadata = get_class_metadata(node, h)
                        if metadata:
                            class_metadata[base_class][node.spelling] = metadata
                            if 'registry_name' in metadata:
                                print(f"    Registry name: {metadata['registry_name']}")
                            if 'aliases' in metadata:
                                print(f"    Aliases: {', '.join(metadata['aliases'])}")
                        
                        # Get constructors for this subclass
                        subclass_constructors_list = get_class_constructors(node, node.spelling)
                        subclass_constructors[base_class][node.spelling] = subclass_constructors_list
                        
                        # Store subclass with template info
                        found_classes[base_class].append((node.spelling, h, subclass_template_params))

    # Write output file
    with open(output_path, "w") as f:
        write_includes(f, base_classes, base_classes_found, found_classes)
        write_lookup_functions(f, base_classes, base_class_categories, found_classes, class_metadata)
        write_combined_lookup_function(f, set(base_class_categories.get(bc, bc.lower()) for bc in base_classes))
        write_category_lookup(f, base_classes, base_class_categories)
        write_constructor_dependencies(f, base_classes, found_classes, subclass_constructors)
        write_constructor_signatures(f, base_classes, found_classes, subclass_constructors)
        write_print_constructor_help(f)
        write_class_hierarchy_functions(f, base_classes, found_classes)
        write_type_identification_functions(f, base_classes, template_parameters, found_classes)
        write_config_param_cast_helper(f)
        write_factory_functions(f, base_classes, template_parameters, base_class_categories,
                                found_classes, subclass_constructors)


if __name__ == "__main__":
    generate()
