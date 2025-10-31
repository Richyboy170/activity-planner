# INSTALLATION GUIDE - Where to Put the Files

## YOUR PROJECT STRUCTURE:

```
C:\Users\HP\Desktop\activity-website\
├── app_optimized.py
├── models/
│   ├── bm25_docs.pkl
│   ├── embeddings.npy
│   ├── faiss_index.bin
│   └── activities_processed.csv
├── static/                    ← PUT FILES HERE
│   ├── api-integration.js
│   └── results-page-styles.css
├── templates/
│   └── index.html
└── venv/
```

---

## FILE 1: api-integration.js

### WHAT IT IS:
Enhanced JavaScript with results page display logic

### WHERE TO PUT IT:
```
C:\Users\HP\Desktop\activity-website\static\api-integration.js
```

### STEPS:
1. Download `api-integration.js` from the files above
2. Navigate to: `C:\Users\HP\Desktop\activity-website\static\`
3. **DELETE your old** `api-integration.js` file
4. **PASTE the new** `api-integration.js` file here

### VERIFY:
- File should be in: `static/api-integration.js`
- File size should be around 9-10 KB

---

## FILE 2: results-page-styles.css

### WHAT IT IS:
Beautiful CSS styling for the results page

### TWO OPTIONS:

#### OPTION A: Separate CSS File (Recommended for cleanliness)
1. Download `results-page-styles.css` from the files above
2. Navigate to: `C:\Users\HP\Desktop\activity-website\static\`
3. **PASTE** `results-page-styles.css` here
4. In your `index.html`, add this line in the `<head>` section:
   ```html
   <link rel="stylesheet" href="/static/results-page-styles.css">
   ```

#### OPTION B: Embed in index.html (If you prefer single file)
1. Download `results-page-styles.css`
2. Open your `index.html` file in VSCode
3. Find the `<style>` tag in the `<head>` section
4. Copy ALL the content from `results-page-styles.css`
5. Paste it into your `<style>` tag (before `</style>`)

### VERIFY:
- File should be in: `static/results-page-styles.css` (if Option A)
- OR embedded in `index.html` (if Option B)
- File size should be around 6-7 KB

---

## FILE 3: Update index.html

### REQUIRED CHANGE:
Wrap your main page content with a `<div id="main-page">` tag

### WHAT TO DO:

1. Open your `index.html` file
2. Find where your main content starts (usually right after `<body>`)
3. Wrap everything with:
   ```html
   <div id="main-page">
       <!-- ALL YOUR EXISTING CONTENT GOES HERE -->
   </div>
   ```

### EXAMPLE:

**BEFORE:**
```html
<body>
    <!-- Your content -->
    <div class="header">...</div>
    <div class="main-container">...</div>
    <!-- More content -->
</body>
```

**AFTER:**
```html
<body>
    <div id="main-page">
        <!-- Your content -->
        <div class="header">...</div>
        <div class="main-container">...</div>
        <!-- More content -->
    </div>
</body>
```

### ADD CSS LINK (if using Option A):

Find the `<head>` section and add:
```html
<head>
    <!-- ... existing links ... -->
    <link rel="stylesheet" href="/static/results-page-styles.css">
    <!-- ... rest of head ... -->
</head>
```

---

## QUICK CHECKLIST:

- [ ] Downloaded `api-integration.js`
- [ ] Placed in `static/api-integration.js`
- [ ] Downloaded `results-page-styles.css`
- [ ] Placed in `static/results-page-styles.css` OR embedded in `index.html`
- [ ] Added `<link>` tag in `index.html` (if separate CSS file)
- [ ] Wrapped main content with `<div id="main-page">` in `index.html`
- [ ] Saved all files

---

## FINAL STEPS:

1. **Restart Flask Backend:**
   ```bash
   cd C:\Users\HP\Desktop\activity-website
   python app_optimized.py
   ```

2. **Refresh Browser:**
   - Go to http://127.0.0.1:5000
   - Hard refresh: Ctrl+F5

3. **Test:**
   - Add group members
   - Click "Get Recommendations" button
   - Should see results page with:
     - Group profile
     - Member badges
     - Ranked activity cards
     - Back to Home button

---

## IF SOMETHING GOES WRONG:

### Results page doesn't appear:
- Check: Browser console (F12) for errors
- Check: id="main-page" exists in HTML
- Check: Flask is running (terminal should show logs)

### Styling looks broken:
- Clear cache: Ctrl+Shift+Delete
- Hard reload: Ctrl+F5

### Check Flask logs for API errors:
- Look at terminal where Flask is running
- Should show POST requests to `/api/recommend`

---

## DONE! 🎉

Your Family Activity Planner now has a beautiful results page showing:
- Group profile with member details
- Ranked activities (1st, 2nd, 3rd...)
- Match scores with color coding
- Ability, age, and interest matching
- Back to home functionality