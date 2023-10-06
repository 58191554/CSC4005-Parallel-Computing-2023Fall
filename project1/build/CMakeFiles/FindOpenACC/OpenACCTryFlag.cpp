
int main(){
#ifdef _OPENACC
  return 0;
#else
  breaks_on_purpose
#endif
}
