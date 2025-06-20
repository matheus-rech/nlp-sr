# Otto-SR Interface Preview

## 🖼️ Abstract Navigator Component Preview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 📊 Screening Metrics                                                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│   Total        Screened      Included      Excluded      Conflicts    Avg Confidence│
│    476           320           125           180            15           85%        │
│               [▓▓▓▓▓▓▓░░░]                                                         │
│                  67.2%         39.1%         56.3%         4.7%                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [<] 125 of 320 [>]                              Filter: [All Citations ▼]           │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────┬───────────────────────────────────────────┐
│ 📄 Citation Details                     │ 🧠 AI Evaluation                          │
│                                         │                                           │
│ Effect of mindfulness-based stress      │ [Summary] [AI 1] [AI 2]                   │
│ reduction on anxiety in adults          │                                           │
│                                         │ Final Decision: [✓ Included]              │
│ 👥 Smith J, Johnson M, Williams K      │ Confidence: 92%                           │
│ 📚 Journal of Anxiety Research          │ Consensus: agree_include                  │
│ 📅 2023                                 │                                           │
│                                         │ PICO Matches:                             │
│ Abstract:                               │ ✓ Population  ✓ Intervention              │
│ This study examined the effectiveness   │ ✓ Comparator  ✓ Outcome                  │
│ of mindfulness-based stress reduction   │ ✓ Timeframe   ✓ Study Type               │
│ (MBSR) on anxiety levels in adults.    │                                           │
│ A randomized controlled trial was       │ Conservative AI (GPT-4):                  │
│ conducted with 120 participants aged    │ Decision: Include (95%)                   │
│ 18-65 years. Results showed significant│ "Clear RCT with appropriate population"   │
│ reduction in anxiety scores...          │                                           │
│                                         │ Liberal AI (GPT-3.5):                     │
│ Keywords:                               │ Decision: Include (89%)                   │
│ [mindfulness] [anxiety] [RCT] [adults]  │ "Meets all inclusion criteria"            │
│                                         │                                           │
└─────────────────────────────────────────┴───────────────────────────────────────────┘
```

## 🎨 Key Visual Elements

### 1. **Metrics Dashboard**
- **Visual Progress Bar**: Shows screening completion
- **Color-Coded Stats**: Green (included), Red (excluded), Yellow (conflicts)
- **Real-time Updates**: Numbers update as screening progresses

### 2. **Navigation Controls**
- **Previous/Next Buttons**: Easy sequential navigation
- **Position Indicator**: "125 of 320" shows current position
- **Filter Dropdown**: Quick access to subsets

### 3. **Citation Display Panel**
- **Structured Metadata**: Icons for authors, journal, year
- **Scrollable Abstract**: Full text with scroll for long abstracts
- **Keyword Tags**: Visual badges for quick topic identification

### 4. **AI Evaluation Panel**
- **Tabbed Interface**: Summary, AI 1, AI 2 views
- **Decision Badge**: Color-coded visual indicator
- **PICO Grid**: Checkmarks/crosses for criteria matches
- **Confidence Meter**: Percentage with color coding
- **Evidence Quotes**: Key supporting text from abstract

## 🔄 Interactive Features

### User Interactions:
1. **Click Navigation**: Previous/Next buttons
2. **Keyboard Shortcuts**: Arrow keys for navigation
3. **Filter Selection**: Dropdown to view subsets
4. **Tab Switching**: Click tabs to see detailed AI reasoning
5. **Hover Effects**: Tooltips on badges and metrics

### Real-time Updates:
- Metrics refresh as new screenings complete
- Progress bar animates during batch processing
- Conflict badges appear immediately when detected
- Human review updates reflect instantly

## 📱 Responsive Design

### Desktop View (shown above)
- Side-by-side citation and evaluation panels
- Full metrics dashboard
- All information visible at once

### Tablet View
- Stacked layout with citation above evaluation
- Condensed metrics with expandable details
- Touch-optimized navigation

### Mobile View
- Single column layout
- Swipe navigation between citations
- Collapsible sections for space efficiency
- Bottom navigation bar

## 🎯 User Experience Highlights

1. **Information Hierarchy**: Most important info (decision, confidence) prominently displayed
2. **Visual Feedback**: Loading states, hover effects, transition animations
3. **Accessibility**: High contrast, clear icons, keyboard navigation
4. **Performance**: Virtual scrolling for large datasets
5. **Export Options**: Quick export buttons for current view

## 💡 Additional Features in Full Implementation

- **Search Bar**: Find specific citations by title/author
- **Bulk Actions**: Select multiple citations for batch operations
- **Comparison View**: Side-by-side display of conflicting evaluations
- **Notes Panel**: Add human reviewer comments
- **History Timeline**: Track all actions on each citation
- **Statistics Charts**: Visual representation of screening progress

This interface design prioritizes efficiency and clarity, helping researchers quickly navigate through citations while having all relevant information immediately accessible for informed decision-making.