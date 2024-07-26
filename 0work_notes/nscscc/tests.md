# Tests
## Stream Test
### Spec
Addr Pointers:
| Register | Addr |
| --- | --- |
| a0 | h8010_0000 |
| a1 | h8040_0000 |
| a2 | h8040_0000 |

Load `h8010_0000 - h8040_0000` word by word into `h8040_0000 - h8070_0000`
5 instruction per loop:
```bash
stream_next:
    ld.w        t0,a0,0x0
    st.w        t0,a1,0x0
    addi.w      a0,a0,0x4
    addi.w      a1,a1,0x4
    bne         a0,a2,stream_next
```
### Optimization
#### Bottlenecks:
- Structure Hazard
  - BaseRam should be Read from Instruction Channel and Data Channel
- Branch Prediction
  - Backward Branch takes up the most cases
- Load/Store Cycles
  - Consequent Load-Store

#### Approaches:
- An Icache to minimize the times of accessing the bus, making the bus works for Data Channel only.This will work well for multi-cycle memory access, Single cycle may not work so well.
- BTFNT is enough

## Matrix
```C
void matrix(int a[128][128],int b[128][128],int c[128][128],unsigned int n) {
            unsigned int i,j,k;
            for (k=0; k!=n; k++) {
                for (i=0; i!=n; i++) {
                    int r = a[i][k];
                    for (j=0; j!=n; j++)
                    // Consequent Load and Store
                        c[i][j] += r * b[k][j];
                }
            }
        }
```


### Optimization
#### Bottleneck
- Consequent Load
- consequent Store
#### Approaches

## Cryptonight 
```C
void crn(int pad[],unsigned int a,unsigned int b,unsigned int n) {
            unsigned int k;
            // 1. Consequent Store into 0x8040_0000
            for (k=0; k!=0x80000; k++)
                pad[k] = k;

            for (k=0; k!=n; k++) {
                unsigned int t, addr1, addr2;
                addr1 = a & 0x7FFFF;
                // Load from a Random addr
                t = (a >> 1) ^ (pad[addr1] << 1); // Replace the AES step
                // Store to the same addr
                pad[addr1] = t ^ b;
                addr2 = t & 0x7FFFF;
                b = t;
                // Load from another random addr
                t = pad[addr2];
                a += b * t;
                // Store to the same addr
                pad[addr2] = a;
                a ^= t;
            } 
        }
```


### Optimization
#### Bottleneck
To see further.pick out the assembly.
```C
// t0=0x80000 
fill_next:
    st.w        t3,t4,0
    addi.w      t3,t3,1
    addi.w      t4,t4,4
    bne         t3,t0,fill_next
```
$2^{19}$ Store Loop. It may be useful to coerse the writes

```C
// a3=0x100000
crn_hext:
    and         t0,a1,t2
    slli.w      t0,t0,2
    add.w       t0,a0,t0
    // *****************
    ld.w        t3,t0,0
    srli.w      t4,a1,1
    slli.w      t3,t3,1
    xor         t3,t3,t4
    and         t4,t3,t2
    xor         a2,t3,a2
    slli.w      t4,t4,2
    st.w        a2,t0,0
    add.w       t4,a0,t4
    // *****************
    ld.w        t0,t4,0
    or          a2,zero,t3
    mul.w       t3,t3,t0
    addi.w      t1,t1,1
    add.w       a1,t3,a1
    st.w        a1,t4,0
    xor         a1,t0,a1
    bne         a3,t1,crn_hext
```
$2^{20}$ crn Loop.

#### Approaches
- BTFNT works well for this situation