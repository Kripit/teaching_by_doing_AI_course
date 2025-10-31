"""
Fix LaTeX formulas in README files to be GitHub-compatible.
GitHub supports math rendering using $...$ for inline and $$...$$ for blocks.
"""

import os
import re

def fix_math_in_file(filepath):
    """Fix math formulas in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # GitHub requires blank lines before and after $$ blocks
    # Pattern: Find $$ blocks and ensure they have blank lines
    
    # Add blank line before $$
    content = re.sub(r'([^\n])\n\$\$', r'\1\n\n$$', content)
    
    # Add blank line after $$
    content = re.sub(r'\$\$\n([^\n])', r'$$\n\n\1', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed: {filepath}")

def main():
    """Find and fix all README files."""
    base_dir = os.path.dirname(__file__)
    modules_dir = os.path.join(base_dir, 'modules')
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(modules_dir):
        for file in files:
            if file == 'README.md':
                filepath = os.path.join(root, file)
                fix_math_in_file(filepath)
                fixed_count += 1
    
    # Fix main README too
    main_readme = os.path.join(base_dir, 'README.md')
    if os.path.exists(main_readme):
        fix_math_in_file(main_readme)
        fixed_count += 1
    
    print(f"\n🎉 Fixed {fixed_count} README files!")
    print("\nGitHub Math Rendering Tips:")
    print("1. Use $...$ for inline math: $x^2$")
    print("2. Use $$...$$ for block math with blank lines before/after")
    print("3. Complex formulas might still not render - use code blocks as fallback")

if __name__ == "__main__":
    main()
