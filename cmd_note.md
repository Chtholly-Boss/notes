# Command Notes
## Table of commands
| command | Description |
| --- | --- |
| [read](#read-read) | read info from stdin |
| [tmux](#tmux-tmux) | terminal multiplexer |
### read {#Read}
[Online Tutorial of `read` command](https://phoenixnap.com/kb/bash-read)
common uses:
```shell
read var # read a line and assign to var
read -p "prompt" var # echo prompt and read 
read -t 5 -p "enter your name" name # if read nothing in 5 seconds,return non-zero
read -n 1 -p "Do you want to continue? [Y/N]" answer # specify chars to read
# read from files
cat $HOME/foo.txt | while read line # if nothing to read,return non-zero
do
    ...
done
```
### tmux {#Tmux}
[Online tutorial of `tmux`](https://louiszhai.github.io/2017/09/30/tmux/)
- prefix with `ctrl + b`
- C-b is for `ctrl + b`
- S-up is for `shift + up`
- M-1 is for `alt + 1`

|type | shortcut | function |
| --- | --- | --- |
| move | (in shell) tmux at | attach |
| move | `d` | detach |
| move | `ctrl + up/down/left/right` | select to the pane up/donw/left/right | 
| move | `num` | select window "num" |
| split | `ctrl + "` | split vertically |
| split | `ctrl + %` | split horizontally |
| resize | `ctrl + up/down/left/right` | resize up/down/left/right |
| delete | `ctrl + &` | kill current window |
| delete | `ctrl + x` | kill current pane | 
| create | `ctrl + c` | create new window |
``` shell
# common commands
```
