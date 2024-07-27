#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/param.h"
char* getline();
int main(int argc,char* argv[]) {
    if (argc < 1) {
        fprintf(2,"Usage: xargs command [arguments...]");
        exit(1);
    }
    char* cmdArgv[MAXARG];
    for (int i = 1; i < argc; i++) {
        cmdArgv[i-1] = argv[i];
    }
    for(;;) {
        cmdArgv[argc-1] = getline();
        if(strcmp(cmdArgv[argc-1],"") == 0) break;
        if (fork() == 0)
        {   
            exec(cmdArgv[0],cmdArgv);
        } else {
            wait((int* ) 0);
        }
    }
    exit(0);
}

char* getline() {
    static char buf[512];
    buf[0] = 0;
    int id = 0;
    char s;
    while(read(0,&s,1) > 0) {
        if(s == '\n') break;
        buf[id++] = s;
    }
    buf[id] = '\0';
    return buf;
}
