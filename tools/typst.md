# Typst
## Basic Editting
- Headers : `= Header`
- Italian : `_Italian_`
- List
  - Ordered : ` + order list `
  - Unordered : ` - unordered list `
- Figure
```
#figure(
  image("glacier.jpg", width: 70%),
  caption: [
    caption
  ]
) <label>
```

- Math : `$ Math Equation $` 
- Set Rule : `#set par(justify: true)` --- following content will be aligned
- type `#` and `ctrl + space` to open auto-complete plane
- common set rules:
  - text
  - page
  - par
  - heading
  - document
## Define Functions
```
// Defining Functions
#let name(param: default,...) = {}
// Unnamed Function : like anonymous function
it => [#it #it]
```

## issues
- inside a function,# sometimes not needed,but sometimes needed
## Useful Packages
[Useful Package to display different kinds of information](https://typst.app/universe/package/gentle-clues)