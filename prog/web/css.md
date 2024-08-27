# CSS
This is my learning notes of **Cascading Style Sheets**.

## Reference
* [MDN-CSS-Reference](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference)

## Basis
```css
/* The basic rule in css */
selector {
  property: value
  ...
}

/* You can specify a list of selectors */
selector_1,
selector_2,
... {
  /* This will apply the property to all the element selected*/
  property: value
}
/*  But the whole rule will be omitted
    if one of the selectors is invalid
*/
```
### Selectors
type selector: `tag`
  : selects an HTML tag/element

The universal selector: `*`
  : selects everything

class selector: `.class`
  : select everything in the document with that class applied to it
    * add a `class` attribute to the element in HTML
    * To specify an element of the class, use `tag.class`
      * i.e. `p.highlight` will select `p` elements with `highlight` class.
    * To select more than one class, use `.class1.class2`

ID selector: `#id`
  : select an element that has the `id` set on it
    * an ID can be used only once per page
    * elements can only have a single id value applied to them.

Presence and value selectors: `tag[attr=value]`
  : select an element based on the presence of an attribute alone, or on various different matches against the value of the attribute.
    * refer to [MDN-Attribute-Selector](https://developer.mozilla.org/en-US/docs/Learn/CSS/Building_blocks/Selectors/Attribute_selectors) for more details

Pseudo class: `:pseudoClass`
  : A pseudo-class is a selector that selects elements that are in a specific state
    * examples: 
      * `a:hover` select the `a` element when user point to it
      * `article p:first-child`: select the first paragraph in the `article` class

Pseudo elements: `::pseudoElement`
  : act as if you had added a whole new HTML element into the markup, rather than applying a class to existing elements.
    * examples:
      * `article p::first-line` select the **first line** in the `article` class

descendant combinator: `tagA tagB`
  : match `tagA` first, then match `tagB`
    * example: `.box p` select paragraph with `box` class

child combinator: `parent > directChild`
  :  It matches only those elements matched by the second selector that are the direct children of elements matched by the first.

next-sibling combinator: `cur + next`
  : It matches only those elements matched by the second selector that are the next sibling element of the first selector
    * example: 
      * suppose in html, `<h1>foo</h1> <p> bar </p> <p> foobar </p>`
      * to select `bar`, `h1 + p`

subsequent-sibling combinator: `cur ~ subsequent`
  : select siblings of an element even if they are not directly adjacent
    * in the above example
    * `h1 ~ p` select `bar` and `foobar`

```css
/* YOu can nest combinators */
p {
  ~ img {
  }
}
/* This is parsed by the browser as */
p ~ img {
}

p {
  & img {
  }
}
/* This is parsed by the browser as */
p img {
}

```
