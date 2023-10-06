
#include <stdio.h>
const char accver_str[] = { 'I', 'N', 'F', 'O', ':', 'O', 'p', 'e', 'n', 'A',
                            'C', 'C', '-', 'd', 'a', 't', 'e', '[',
                            ('0' + ((_OPENACC/100000)%10)),
                            ('0' + ((_OPENACC/10000)%10)),
                            ('0' + ((_OPENACC/1000)%10)),
                            ('0' + ((_OPENACC/100)%10)),
                            ('0' + ((_OPENACC/10)%10)),
                            ('0' + ((_OPENACC/1)%10)),
                            ']', '\0' };
int main()
{
  puts(accver_str);
  return 0;
}
