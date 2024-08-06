# File System
The purpose of a file system is to organize and store data.
Challenges:
* needs on-disk data structures to:
  * represent the tree of named directories and files
  * record identities of the blocks that hold each file’s content
  * record which areas of the disk are free
* support **crash recovery**
  *  if a crash (e.g., power failure) occurs, the file system must still work correctly after a restart.
* deal with concurrency(different processes operate on the file system at the same time)
* maintain an in-memory cache of popular blocks to accelerate

## Overview
Layers of the xv6 file system (bottom-up)
* disk
* buffer cache
* logging
* inode
* directory
* pathname
* file descriptor

Disk hardware traditionally presents the data on the disk as a numbered sequence of 512-byte blocks (also called **sectors**)

layout in the disk:
| boot | super | log | Inodes | bit map | data ... data |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 1 | 2 | - | - | - | - | 

## Buffer cache Layer
Jobs:
* synchronize access to disk blocks to ensure
  * only one copy of a block is in memory 
  * only one kernel thread at a time uses that copy;
* cache popular blocks so that they don’t need to be re-read from the slow disk

overview:
* per-buffer sleep-lock
* `bread` returns a locked buffer to be read or modified
* `bwrite` writes a modified buffer to the appropriate block on the disk. 
* A kernel thread must release a buffer by calling `brelse` when it is done with it. 

## Code: Buffer cache
* structure: double-linked list of buffers
  * `binit` initialize the buffers
  * all other access via `bcache.head`
* fields in a buffer:
  * `valid`: the buffer contains a copy of a disk block
  * `disk`: contents of the buffer has been handed to the disk
* `bread`: first call `bget`, if the buffer gotten with `valid=0`, read from the disk.
* locks being used:
  * The `sleep-lock` protects reads and writes of the block’s buffered content
  * `bcache.lock` protects information about which blocks are cached.

## Logging Layer

## Log Design

## Code: Logging

## Code: Block Allocator

## Inode Layer

## Code: Inodes

## Code: Inode Content

## Code: Directory Layer

## Code: Path names

## File Descriptor Layer

## Code: System Calls

## Real World