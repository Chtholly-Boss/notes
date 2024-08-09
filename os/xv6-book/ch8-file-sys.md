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
Problem related: Crash Recovery.
Depending on **the order of the disk writes**, the crash may:
* leave an inode with a reference to a content block marked free
* leave an allocated but unreferenced content block

An xv6 system call does not directly write the on-disk file system data structures. Instead, it places **a description of all the disk writes** it wishes to make **in a log on the disk**.
* the system logged all of its writes
* writes a special commit record to the disk
* start to `truly` write
* when finished normally, erase the log on the disk

## Log Design
The log resides at a known fixed location, specified in the **superblock**.
It consists of a header block followed by a sequence of updated block copies (`logged blocks`). 

Group Commit: committing several transactions together.(or `batching`)

Xv6 dedicates **a fixed amount of space on the disk** to hold the log. The total number of blocks written by the system calls in a transaction must fit in that space
* No single system call can be allowed to write more distinct blocks than there is space in the log
* the logging system cannot allow a system call to start unless it is certain that the system call’s writes will fit in the space remaining in the log

## Code: Logging

## Code: Block Allocator

## Inode Layer

## Code: Inodes

## Code: Inode Content

## Code: Directory Layer
```C
// Directory is a file containing a sequence of dirent structures.
#define DIRSIZ 14

struct dirent {
  ushort inum;
  char name[DIRSIZ];
};
// Look for a directory entry in a directory.
// If found, set *poff to byte offset of entry.
struct inode*
dirlookup(struct inode *dp, char *name, uint *poff)
{
  uint off, inum;
  struct dirent de;

  if(dp->type != T_DIR)
    panic("dirlookup not DIR");

  for(off = 0; off < dp->size; off += sizeof(de)){
    if(readi(dp, 0, (uint64)&de, off, sizeof(de)) != sizeof(de))
      panic("dirlookup read");
    if(de.inum == 0)
      continue;
    if(namecmp(name, de.name) == 0){
      // entry matches path element
      if(poff)
        *poff = off;
      inum = de.inum;
      return iget(dp->dev, inum);
    }
  }

  return 0;
}

// Write a new directory entry (name, inum) into the directory dp.
// Returns 0 on success, -1 on failure (e.g. out of disk blocks).
int
dirlink(struct inode *dp, char *name, uint inum)
{
  int off;
  struct dirent de;
  struct inode *ip;

  // Check that name is not present.
  if((ip = dirlookup(dp, name, 0)) != 0){
    iput(ip);
    return -1;
  }

  // Look for an empty dirent.
  for(off = 0; off < dp->size; off += sizeof(de)){
    if(readi(dp, 0, (uint64)&de, off, sizeof(de)) != sizeof(de))
      panic("dirlink read");
    if(de.inum == 0)
      break;
  }

  strncpy(de.name, name, DIRSIZ);
  de.inum = inum;
  if(writei(dp, 0, (uint64)&de, off, sizeof(de)) != sizeof(de))
    return -1;

  return 0;
}
```
## Code: Path names
```C
// return an inode of a path
struct inode*
namei(char *path)
{
  char name[DIRSIZ];
  return namex(path, 0, name);
}

struct inode*
nameiparent(char *path, char *name)
{
  return namex(path, 1, name);
}
// generalized form: namex
// Look up and return the inode for a path name.
// If parent != 0, return the inode for the parent and copy the final
// path element into name, which must have room for DIRSIZ bytes.
// Must be called inside a transaction since it calls iput().
static struct inode*
namex(char *path, int nameiparent, char *name)
{
  struct inode *ip, *next;

  if(*path == '/')
    ip = iget(ROOTDEV, ROOTINO);
  else
    ip = idup(myproc()->cwd);

  while((path = skipelem(path, name)) != 0){
    ilock(ip);
    if(ip->type != T_DIR){
      iunlockput(ip);
      return 0;
    }
    if(nameiparent && *path == '\0'){
      // Stop one level early.
      iunlock(ip);
      return ip;
    }
    if((next = dirlookup(ip, name, 0)) == 0){
      iunlockput(ip);
      return 0;
    }
    iunlockput(ip);
    ip = next;
  }
  if(nameiparent){
    iput(ip);
    return 0;
  }
  return ip;
}
```
## File Descriptor Layer
```C
// see file.h
struct file {
  enum { FD_NONE, FD_PIPE, FD_INODE, FD_DEVICE } type;
  int ref; // reference count
  char readable;
  char writable;
  struct pipe *pipe; // FD_PIPE
  struct inode *ip;  // FD_INODE and FD_DEVICE
  uint off;          // FD_INODE
  short major;       // FD_DEVICE
};
// functions see file.c
// file.c
struct file*    filealloc(void);
void            fileclose(struct file*);
struct file*    filedup(struct file*);
void            fileinit(void);
int             fileread(struct file*, uint64, int n);
int             filestat(struct file*, uint64 addr);
int             filewrite(struct file*, uint64, int n);
```
## Code: System Calls
```C
// Create the path new as a link to the same inode as old.
uint64
sys_link(void)
{
  char name[DIRSIZ], new[MAXPATH], old[MAXPATH];
  struct inode *dp, *ip;

  if(argstr(0, old, MAXPATH) < 0 || argstr(1, new, MAXPATH) < 0)
    return -1;

  begin_op();
  if((ip = namei(old)) == 0){
    end_op();
    return -1;
  }

  ilock(ip);
  if(ip->type == T_DIR){
    iunlockput(ip);
    end_op();
    return -1;
  }

  ip->nlink++;
  iupdate(ip);
  iunlock(ip);

  if((dp = nameiparent(new, name)) == 0)
    goto bad;
  ilock(dp);
  if(dp->dev != ip->dev || dirlink(dp, name, ip->inum) < 0){
    iunlockput(dp);
    goto bad;
  }
  iunlockput(dp);
  iput(ip);

  end_op();

  return 0;

bad:
  ilock(ip);
  ip->nlink--;
  iupdate(ip);
  iunlockput(ip);
  end_op();
  return -1;
}
```

```C
static struct inode*
create(char *path, short type, short major, short minor)
{
  struct inode *ip, *dp;
  char name[DIRSIZ];

  if((dp = nameiparent(path, name)) == 0)
    return 0;

  ilock(dp);

  if((ip = dirlookup(dp, name, 0)) != 0){
    iunlockput(dp);
    ilock(ip);
    if(type == T_FILE && (ip->type == T_FILE || ip->type == T_DEVICE))
      return ip;
    iunlockput(ip);
    return 0;
  }

  if((ip = ialloc(dp->dev, type)) == 0){
    iunlockput(dp);
    return 0;
  }

  ilock(ip);
  ip->major = major;
  ip->minor = minor;
  ip->nlink = 1;
  iupdate(ip);

  if(type == T_DIR){  // Create . and .. entries.
    // No ip->nlink++ for ".": avoid cyclic ref count.
    if(dirlink(ip, ".", ip->inum) < 0 || dirlink(ip, "..", dp->inum) < 0)
      goto fail;
  }

  if(dirlink(dp, name, ip->inum) < 0)
    goto fail;

  if(type == T_DIR){
    // now that success is guaranteed:
    dp->nlink++;  // for ".."
    iupdate(dp);
  }

  iunlockput(dp);

  return ip;

 fail:
  // something went wrong. de-allocate ip.
  ip->nlink = 0;
  iupdate(ip);
  iunlockput(ip);
  iunlockput(dp);
  return 0;
}
```

```C
uint64
sys_open(void)
{
  char path[MAXPATH];
  int fd, omode;
  struct file *f;
  struct inode *ip;
  int n;

  argint(1, &omode);
  if((n = argstr(0, path, MAXPATH)) < 0)
    return -1;

  begin_op();

  if(omode & O_CREATE){
    ip = create(path, T_FILE, 0, 0);
    if(ip == 0){
      end_op();
      return -1;
    }
  } else {
    if((ip = namei(path)) == 0){
      end_op();
      return -1;
    }
    ilock(ip);
    if(ip->type == T_DIR && omode != O_RDONLY){
      iunlockput(ip);
      end_op();
      return -1;
    }
  }

  if(ip->type == T_DEVICE && (ip->major < 0 || ip->major >= NDEV)){
    iunlockput(ip);
    end_op();
    return -1;
  }

  if((f = filealloc()) == 0 || (fd = fdalloc(f)) < 0){
    if(f)
      fileclose(f);
    iunlockput(ip);
    end_op();
    return -1;
  }

  if(ip->type == T_DEVICE){
    f->type = FD_DEVICE;
    f->major = ip->major;
  } else {
    f->type = FD_INODE;
    f->off = 0;
  }
  f->ip = ip;
  f->readable = !(omode & O_WRONLY);
  f->writable = (omode & O_WRONLY) || (omode & O_RDWR);

  if((omode & O_TRUNC) && ip->type == T_FILE){
    itrunc(ip);
  }

  iunlock(ip);
  end_op();

  return fd;
}
```
## Real World
* buffer caches are different
  * purposes: caching and synchronizing access to the disk
* logging system more efficient
* different crash recovery ways
* disk failures
* storage requirements
* snapshots
* incremental backup