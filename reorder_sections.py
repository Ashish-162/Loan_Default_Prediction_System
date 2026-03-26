import re

# Read the dashboard.html file
with open('dashboard.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and extract the two sections - be more flexible with whitespace
# Real-Time Dataset Metrics section
realtime_start = content.find('Real-Time Dataset Metrics</h2>')
if realtime_start == -1:
    print("Could not find Real-Time Dataset Metrics header")
else:
    # Go back to find the opening div
    realtime_div_start = content.rfind('<div class="section">', 0, realtime_start)
    # Find the closing of this section (next closing </div></div>)
    realtime_end = content.find('</div>\n                </div>', realtime_start) + len('</div>\n                </div>')
    realtime_section = content[realtime_div_start:realtime_end]
    print(f"Found Real-Time section ({len(realtime_section)} chars)")

# Current Input Snapshot section
snapshot_start = content.find('Current Input Snapshot</h2>')
if snapshot_start == -1:
    print("Could not find Current Input Snapshot header")
else:
    # Go back to find the opening div
    snapshot_div_start = content.rfind('<div class="section">', 0, snapshot_start)
    # Find the closing
    snapshot_end = content.find('</div>\n                </div>', snapshot_start) + len('</div>\n                </div>')
    snapshot_section = content[snapshot_div_start:snapshot_end]
    print(f"Found Snapshot section ({len(snapshot_section)} chars)")

if realtime_start != -1 and snapshot_start != -1:
    # Remove both sections
    temp_content = content[:realtime_div_start] + content[realtime_end:]
    # Adjust snapshot positions after removal
    offset = len(content) - len(temp_content)
    snapshot_div_start -= offset
    snapshot_end -= offset
    
    temp_content = temp_content[:snapshot_div_start] + temp_content[snapshot_end:]
    
    # Insert snapshot first, then realtime at the position where realtime was
    realtime_pos = temp_content.find('Real-Time Model Metrics')
    if realtime_pos != -1:
        # Find the start of that section
        section_start = temp_content.rfind('<div class="section">', 0, realtime_pos)
        # Insert both sections before this
        new_content = temp_content[:section_start] + snapshot_section + '\n\n                    ' + realtime_section + '\n\n                    ' + temp_content[section_start:]
        
        with open('dashboard.html', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✓ Section order updated successfully!")
    else:
        print("Could not find insertion point")

