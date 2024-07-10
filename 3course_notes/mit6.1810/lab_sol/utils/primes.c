#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
int fork2filt(int* p);
int main(int argc,char* argv[]) {
    int p[2];
    pipe(p);
    if (fork() == 0)
    {
        fork2filt(p);
    } else {
        close(p[0]);
        for (int i = 2; i < 36; i++) {
            write(p[1],&i,4);
        }
        close(p[1]);
        // Wait for the main prime process to end
        wait((int*) 0);
    }
    exit (0);
}
int fork2filt(int* p) {
    int prime;
    int buf;
    close(0);
    dup(p[0]);
    close(p[0]);
    close(p[1]);
    if (read(0,&buf,4) > 0) prime = buf;
    else exit(1);

    int pp[2];
    pipe(pp);
    if (fork() == 0) {
        //fprintf(1,"pid: %d\n",getpid());
        fork2filt(pp);
    } else {
        close(pp[0]);
        fprintf(1,"prime %d\n",prime);
        while (read(0,&buf,4) > 0) {
            if (buf % prime == 0) continue;
            write(pp[1],&buf,4);
        }
        close(pp[1]);
        wait((int* ) 0);
    }
    exit(0);
}