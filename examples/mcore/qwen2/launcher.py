import sys
import re
import os
import subprocess
import tempfile
import stat

def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python3 launcher.py <script_path> [key=value ...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    overrides = {}
    for arg in sys.argv[2:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            overrides[key] = value
    
    return script_path, overrides

def apply_overrides(content, overrides):
    new_content = content
    
    # Common alias mapping
    aliases = {
        'gpus': 'ASCEND_RT_VISIBLE_DEVICES',
        'devices': 'ASCEND_RT_VISIBLE_DEVICES',
    }
    
    for key, value in overrides.items():
        # Resolve alias
        key = aliases.get(key, key)
        
        # 1. Try to replace Environment Variable or Bash Variable assignment
        # Matches: export KEY=VAL or KEY=VAL or KEY="VAL"
        # We look for lines starting with (export\s+)?KEY=
        pattern_var = re.compile(r'^(export\s+)?' + re.escape(key) + r'=.*$', re.MULTILINE)
        
        if pattern_var.search(new_content):
            # Preserve 'export ' if it exists
            def replace_func(match):
                prefix = match.group(1) if match.group(1) else ""
                return f'{prefix}{key}="{value}"'
            
            new_content = pattern_var.sub(replace_func, new_content)
            print(f"Updated variable: {key} = {value}")
            continue # If found as variable, we assume that's enough (variables often feed into args)

        # 2. Try to replace Command Line Argument
        # Matches: --key VAL or --key=VAL
        # We handle keys with underscores by trying hyphens too (e.g. learning_rate -> --learning-rate)
        
        # Normalize key for flags: 'learning_rate' -> 'learning-rate'
        flag_key = key.replace('_', '-')
        
        # Pattern matches: whitespace, --flag_key, whitespace/equal, value, whitespace/end
        # Value matching is tricky, usually \S+ but could be quoted. Simplified to \S+
        pattern_arg = re.compile(r'(--' + re.escape(flag_key) + r')(\s+|=)(\S+)')
        
        if pattern_arg.search(new_content):
            new_content = pattern_arg.sub(r'\1\2' + value, new_content)
            print(f"Updated argument: --{flag_key} {value}")
            continue
            
        print(f"Warning: Could not find parameter '{key}' (or --{flag_key}) in script to replace.")

    return new_content

def main():
    script_path, overrides = parse_args()
    
    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found.")
        sys.exit(1)
        
    with open(script_path, 'r') as f:
        content = f.read()
        
    modified_content = apply_overrides(content, overrides)
    
    # Create a temporary file
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.basename(script_path)
    temp_script_path = os.path.join(script_dir, f"tmp_{script_name}")
    
    try:
        with open(temp_script_path, 'w') as f:
            f.write(modified_content)
        
        # Make executable
        st = os.stat(temp_script_path)
        os.chmod(temp_script_path, st.st_mode | stat.S_IEXEC)
        
        print(f"Running modified script: {temp_script_path}")
        print("-" * 40)
        
        # Execute
        subprocess.run([temp_script_path], check=False)
        
    finally:
        # Cleanup
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
            print("-" * 40)
            print(f"Cleaned up temporary script: {temp_script_path}")

if __name__ == "__main__":
    main()
