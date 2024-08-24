# Structure of a Website
For a simple plan, see [Planning-a-Simple-Site](https://developer.mozilla.org/en-US/docs/Learn/HTML/Introduction_to_HTML/Document_and_website_structure)

![sample-website](./figure/sample-website.png)

## Semantic Wrappers
* header: `<header>`.
* navigation bar: `<nav>`.
* main content: `<main>`, with various content subsections represented by `<article>`, `<section>`, and `<div>` elements.
* sidebar: `<aside>`; often placed inside `<main>`.
* footer: `<footer>`.

## Non-Semantic Wrappers
* inline: `span`
* block: `div`

```html
<p>
  The King walked drunkenly back to his room at 01:00, the beer doing nothing to
  aid him as he staggered through the door.
  <span class="editor-note">
    [Editor's note: At this point in the play, the lights should be down low].
  </span>
</p>

<div class="shopping-cart">
  <h2>Shopping cart</h2>
  <ul>
    <li>
      <p>
        <a href=""><strong>Silver earrings</strong></a>: $99.95.
      </p>
      <img src="../products/3333-0985/thumb.png" alt="Silver earrings" />
    </li>
    <li>…</li>
  </ul>
  <p>Total cost: $237.89</p>
</div>
```

## Others
* `br`: line break
* `hr`: horizontal rule

```html
<p>
  1st line <br />
  2nd line <br />
</p>

<p>
    ... about A
</p>
<hr />
<p>
    ... about B
</p>
```