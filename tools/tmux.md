# Tmux
Blogs useful to get started:
* [Guide-to-Tmux](https://hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)
* [tmux-conf](https://hamvocke.com/blog/a-guide-to-customizing-your-tmux-conf/)

A Book may be useful: [tmux-2](https://pragprog.com/titles/bhtmux2/tmux-2/)

Repo and Wiki: [tmux-wiki](https://github.com/tmux/tmux/wiki)
## Get Started
```bash
# install
sudo apt-get install tmux
# start
tmux
```
## Basic Operations
`prefix key` + `command key`
* `C-` for `ctrl` 
* `M-` for `meta` or `alt`

`C-b ?` to see a list of available commands

* split panes
    * `C-b %` : left and right
    * `C-b "` : top and bottom
* Navigating Panel: `C-b arrow`
* Close pane: `C-d` or `exit`
* create window: `C-b c`
* Navigating Window
    * `C-b p`: previous
    * `C-b n`: next
    * `C-b number`: a specific window
* detach a session:
    * `C-b d`: detach current session
    * `C-b D`: interactive way 
* re-attach:
    * `tmux ls` to get the list of the sessions created
    * `tmux attach -t session`
* name a session
    * when creating: `tmux new -s name`
    * rename: `tmux rename-session -t old new`
* `C-b z`: zoom in or zoom out

## Common Workflow
```sh
# create a session to work with
tmux new -s name
# create new window when necessary
### in tmux
prefix + , # name the current window
# Split pane when necessary

# when done, detach
prefix + d
# kill the session in bash
tmux kill-session -t name
```
## Configurations
* create a `.tmux.conf` in your home directory
```bash
# configurations
# remap prefix from 'C-b' to 'C-a'
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

# split panes using | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# reload config file (change file location to your the tmux.conf you want to use)
bind r source-file ~/.tmux.conf

# switch panes using Alt-arrow without prefix
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# Enable mouse control (clickable windows, panes, resizable panes)
set -g mouse on

# don't rename windows automatically
set-option -g allow-rename off
```