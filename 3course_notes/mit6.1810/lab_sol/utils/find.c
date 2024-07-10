#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/fs.h"
char* fmtname(char* path);
void find(char* path,char* pattern);
int main(int argc,char* argv[]) {
    if (argc != 3) {
        fprintf(2,"Usage: find dir pattern\n");
        exit(1);
    }
    char *path,*pattern;
    path = argv[1];
    pattern = argv[2];
    find(path,pattern);
    exit(0);
}
void find(char* path,char* pattern) {
    int fd;
    struct stat st;
    char buf[512], *p;
    struct dirent de;
    if((fd = open(path, 0)) < 0){
        fprintf(2, "find: cannot open %s a\n", path);
        return;
    }

    if(fstat(fd, &st) < 0){
        fprintf(2, "find: cannot stat %s\n", path);
        close(fd);
        return;
    }
    switch (st.type) {
        case T_DEVICE:
        case T_FILE:
            if (strcmp(fmtname(path),pattern) == 0) {
                fprintf(1,path);
                fprintf(1,"\n");
            }
            break;
        case T_DIR:
            if(strlen(path) + 1 + DIRSIZ + 1 > sizeof buf){
                printf("find: path too long\n");
                break;
            }
            strcpy(buf, path);
            p = buf+strlen(buf);
            *p++ = '/';
            while(read(fd, &de, sizeof(de)) == sizeof(de)){
                if(de.inum == 0)
                    continue;
                if(strcmp(de.name,".") == 0)
                    continue;
                if(strcmp(de.name,"..") == 0)
                    continue;
                memmove(p, de.name, DIRSIZ);
                p[DIRSIZ] = 0;
                find(buf,pattern);
            }
            break;
    }
    close(fd);
    return;
}
char* fmtname(char* path) {
    static char *p;
    p = "";
    for(p=path+strlen(path); p >= path && *p != '/'; p--)
        ;
    p++;
    return p;
}