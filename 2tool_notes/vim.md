# Vim Tricks
## Split Windows
See this Blog:[Vim: Window Splits](https://dev.to/mr_destructive/vim-window-splits-p3p)
Here are my use cases(Default Key-Bindings):
Prefix: `^W`
- Split vertically / horizontally: `v/s` or commands `vsp/sp`
- Move around: `arrows` or `hjkl`
- Rearranging:
  - Swap two splits(horizontal or vertical): `r`
  - Move splits: `HJKL` or `S-arrow` or `S-hjkl`
  - resize
    - height: `+/-`
    - width: `>/<`
    - equal: `=`
    - using command: `resize {number}/vertical resize {number}`
- Closing
  - Close the Current:`c`
  - Quit all other splits except the current:`o`

## Tabs
See this Blog:[Beginners Guide to Tabs in Vim](https://webdevetc.com/blog/tabs-in-vim/)
Here are my use cases.
- New tab:`tabe(dit) filename`
- Move around(Shortcuts in Normal Mode):
  - move to the next:`gt`
  - move to the previous:`gT`
  - move to a certain:`#gt`,replace the `#` with the position of the tab
- Rearrange
  - move current to end:`tabmove`
  - move to a certain position:`tabmove #`
- Closing
  - Close the current:`tabc(lose)`
  - Quit all other tabs except the current: `tabo(nly)`
