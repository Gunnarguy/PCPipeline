#!/usr/bin/env python3
"""Script to fix indentation issues in processing.py."""

def fix_indentation_issues():
    """Fix the indentation issues in the processing.py file."""
    with open("rag/processing.py", "r") as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix the first problematic section
        if "if suffix in document_exts:" in line and i+1 < len(lines) and "try:" in lines[i+1]:
            fixed_lines.append(line)  # Add the 'if suffix in document_exts:' line
            fixed_lines.append(lines[i+1])  # Add the 'try:' line
            i += 2  # Skip past these two lines
            
            # Now handle the problematic if/else block
            fixed_lines.append("                if HAS_TEXTRACT:\n")
            fixed_lines.append("                    return textract.process(str(path), **self.textract_config).decode('utf-8')\n")
            fixed_lines.append("                else:\n")
            fixed_lines.append("                    print(f\"Cannot process {path}: textract not available\")\n")
            fixed_lines.append("                    return None\n")
            
            # Skip past the bad indentation in the original file
            while i < len(lines) and "except Exception as e:" not in lines[i]:
                i += 1
            
            # Continue normal processing
        
        # Fix the second problematic section
        elif "def _process_generic" in line and "try:" in lines[i+2]:
            # Add lines until we reach the problematic section
            fixed_lines.append(line)
            j = i + 1
            while j < len(lines) and "if HAS_TEXTRACT:" not in lines[j]:
                fixed_lines.append(lines[j])
                j += 1
            i = j  # Update i to current position
            
            # Now handle the problematic if/else block
            fixed_lines.append("            if HAS_TEXTRACT:\n")
            fixed_lines.append("                return textract.process(str(path), **self.textract_config).decode('utf-8')\n")
            fixed_lines.append("            else:\n")
            fixed_lines.append("                print(f\"Cannot process {path}: textract not available\")\n")
            fixed_lines.append("                return None\n")
            
            # Skip past the bad indentation in the original file
            while i < len(lines) and "except Exception as e:" not in lines[i]:
                i += 1
            
            # Continue normal processing
        else:
            fixed_lines.append(line)
            i += 1
    
    # Write the fixed content back to the file
    with open("rag/processing.py", "w") as f:
        f.writelines(fixed_lines)
    
    print("Fixed indentation issues in rag/processing.py")

if __name__ == "__main__":
    fix_indentation_issues()
