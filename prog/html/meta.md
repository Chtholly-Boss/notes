# Metadata in HTML
```html
<!-- a typical html looks like this -->
<!doctype html>
<html lang="en-US">
  <head>
    <meta charset="utf-8" />
    <title>My test page</title>
  </head>
  <body>
    <p>This is my page</p>
  </body>
</html>
```

Head
    : To contain **metadata** about the document

Elements in the **Head**:
* title
* meta
    * name: type of the metadata
    * content
    * icons
    * CSS and JavaScript link

```html
<meta name="author" content="Chris Mills" />
<!-- description may be used in search engine -->
<meta
  name="description"
  content="The MDN Web Docs Learning Area aims to provide
complete beginners to the Web with all they need to know to get
started with developing websites and applications." />

<!-- CSS and JS -->
 <link rel="stylesheet" href="my-css-file.css" />
 <script src="my-js-file.js" defer></script>
```

```html
<!-- to specify the language of the doc -->
<html lang="en-US">
  <!-- also, subsections can have its own language -->
   <p>Japanese example: <span lang="ja">ご飯が熱い。</span>.</p>
</html>
```
