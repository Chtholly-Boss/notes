# Grep
A search tool. 

General Syntax: `grep "pattern" file`

## Learn By Example
* `grep b.g file`
    * matched: lines contain big, bug, bag, buggy, etc.
    * not matched: bg, b12g
    * `.` represents one character
* `grep b.*g file`
    * matched: bg, big, bug, b123g, etc.
    * `*` means repetition (zero possible)
    * `a*` matches a, aa, aaa, ...

Follows just talk about patterns, omitting `grep ... file`

* `bugg\?y`
    * matched: bugy, buggy
    * not matched: bugggy
    * `\?`: zero or one instance
* `Fred\(eric\)\? Smith`
    * matched: Fred Smith, Frederic Smith
    * `\( abc \)`: treated by a single character. 
* `[Hh]ello`
    * matched: Hello, hello
    * `[abc]` means selection from `[ ]`
    * ranges permitted:
        * `[0-3]`: `[0123]`
        * `[a-cA-C]`: `[abcABC]`
        * `[[:alpha:]]`: `[a-zA-Z]`
        * `[[:upper:]]`: `[A-Z]`
        * `[[:lower:]]`: `[a-z]`
        * `[[:digit:]]`: `[0-9]`
        * `[[:alnum:]]`: `[0-9a-zA-Z]`
        * `[[:space:]]`: matches any white space including tabs
* `[[:digit:]]\{3\}[ -]\?[[:digit:]]\{4\}`
    * 7 digit phone numbers
    * `\{ number \}`: repeat `number` times
* `^[[:space:]]*hello[[:space:]]*$`
    * matched: `   hello  `
    * `^` matches the beginning of the line.
    * `$` matches the end of the line
* `I am a \(cat|dog\)`
    * matched: `I am a cat` or `I am a dog`
    * `|` means `or`
## resources
[grep-guide](https://www.panix.com/%7Eelflord/unix/grep.html)