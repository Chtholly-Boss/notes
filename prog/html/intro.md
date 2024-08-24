# Introduction
Element
    : <Opening Tag> Contents </Closing Tag>

```html
<!-- A basic element -->
<p> This is content. </p>

<!-- Nest Element -->
<p>
    My cat is <strong>very</strong> grumpy.
</p>

<!-- void element with attributes -->
 <img
    src="the/path/to/img"
    alt="Alternative text" />

<!-- Boolean attribute -->
 <input type="text" disabled />
<!-- This is equivalent to -->
 <input type="text" disabled="disabled" />
```

```html
<!-- HTML Document -->
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

Character Reference
    : including special **char**s in HTML

| literals | Character Ref |
| --- | --- |
| < | `&lt;` |
| > | `&gt;` |
| " | `&quot;` |
| ' | `&apos;` |
| & | `&amp;` |

```html
<p>In HTML, you define a paragraph using the <p> element.</p>

<p>In HTML, you define a paragraph using the &lt;p&gt; element.</p>
```
