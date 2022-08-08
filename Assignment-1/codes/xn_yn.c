#include <stdio.h>
int main()
{
    FILE *fptr = fopen("xn.txt", "w");
    FILE *fptr_1 = fopen("yn.txt", "w");
    float x[6] = {1, 2, 3, 4, 2, 1};
    int k = 20;
    float y[20];
    y[0] = x[0];
    // y[n] + y[n-1]*0.5 = x[n] + x[n-2]
    y[1] = x[1] - (y[0]*0.5);
    for(int i = 2; i < k; i++)
    {
        if(i < 6)
        {
            y[i] = x[i] + x[i-2] - (0.5*y[i-1]);
        }
        else if((i>=6)&&(i<8))
        {
            y[i] = x[i-2] - (y[i-1]*0.5);
        }
        else
        {
            y[i] = -0.5*y[i-1];
        }
    }
    for(int i = 0; i < 20; i++)
    {
        printf("%f\n", y[i]);
    }
    for (int j = 0; j < 6; j++)
    {
        fprintf(fptr,"%f\n", x[j]);
    }
    for(int k = 0; k < 20; k++)
    {
        fprintf(fptr_1, "%f\n", y[k]);
    }
    fclose(fptr);
    fclose(fptr_1);
}