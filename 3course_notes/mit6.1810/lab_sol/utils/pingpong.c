#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
int main(int argc,char* argv) {
    int pid;
    int p_P2C[2];
    int p_C2P[2];
    char buf[512];
    pipe(p_P2C);
    pipe(p_C2P);

    if (fork() == 0) {
        close(0);
        dup(p_P2C[0]); // Read from p[0]
        close(p_P2C[0]);
        close(p_P2C[1]);
        pid = getpid();
        while (read(0,buf,sizeof buf) > 0) {
            fprintf(1,"%d: received ping\n",pid);
            write(p_C2P[1],buf,1);
        }
    } else {
        pid = getpid();
        close(0);
        dup(p_C2P[0]);
        close(p_C2P[0]);
        close(p_C2P[1]);
        // Start Ping a byte
        write(p_P2C[1],"h",1);
        close(p_P2C[1]);
        if (read(0,buf,sizeof buf) > 0) {
            fprintf(1,"%d: received pong\n",pid);
            //write(p_P2C[1],buf,1);
        }
    }
    exit(0);
}