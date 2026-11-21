
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "helper_math.h"
#include <random>

constexpr unsigned int HEIGHT = 1024;
constexpr unsigned int WIDTH = 1024;
constexpr unsigned int MAX_COLOR = 255;
constexpr unsigned int NUM_SPHERES = 32;

    float mini = -1.0f;
    float maxi = 1.0f;
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> distrib(mini, maxi);  
    std::mt19937 gen_color(43); 
    std::uniform_int_distribution<int> distrib_color(0, 255);   

struct Sphere
{

    float3 pos;
    float radius;
    uchar3 color;
};

struct Ray
{

    float3 origin;
    float3 direction;
    float t;
};

// struct Scene
// {

//     Sphere s;
// };

uchar3 output_buffer[HEIGHT * WIDTH];


__device__ float ray_sphere(Ray r, Sphere sphere){
    
    float3 V = r.origin - sphere.pos;
    float B = 2*dot(r.direction, V);
    float C = dot(V,V) - sphere.radius*sphere.radius;

    float D = B*B - 4*C;

    if(D < 0){

        return 0.0f;
    }

    D = sqrtf(D);
    
    float t1 = (-1.0f*B + D)/2;
    float t2 = (-1.0f*B - D)/2;

    

    if( t1 >  0 && t2 > 0){
      return fminf(t1,t2);
    }else if( t1 > 0 && t2 < 0){
        return t1;
    }else if( t1 <0 && t2 > 0){
        return t2;
    }else{
        return 0.0f;
    }
   
}



__global__ void raytrace(Sphere *sphere,  uchar3 *output) // thread is the ray
{
    Ray ray;
    
    ray.origin = {0, 0, -1}; // camera > screen > object

    float dx = 2.0f/(float)WIDTH;
    float dy = 2.0f/(float)HEIGHT;

    
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;


    ray.direction = {col*dx -1, 1-row*dy, 1}; // convert to canonical

    

    //#TODO: NORMALIZE DIRECTION
    ray.direction = normalize(ray.direction);

    int idx = row * WIDTH + col;
 
    // r_0 + t*r_d 
    // (P - C).(P-C) = R.R
     // r_0^2 + 2*r_0*t*r_d - 2*r_0*C - 2*t*r_d*C + t^2*r_d^2 + C^2 = 0
     //r_d^2*(t^2)  2*r_d*(r_0 - C)*(t) + (r_0-c).(r_0-c) - R^2= 0
     // V = r_0 - C 
     // A = (r_d).(r_d)
     // B = 2*(r_d).V
     // C = V.V - R^2
     // can remove r_d. r_d if normalized because unit vector
     //D = sqrtf(B^2 - 4*C) // check if B^2 - 4*c is not negative
     //t1 = (-b + D)/2
     // t2 = (-b - D)/2
     // if both positive, min (t1,t2)
     //  if only 1 positive, choose positive
     // if both negative, no soln
     float t_min = INFINITY;
     int closest_idx = -1;
     for(int i = 0; i < NUM_SPHERES;i++){

       float t = ray_sphere(ray, sphere[i]);

       if(t > 0 && t< t_min){
        t_min = t;
        closest_idx = i;
       }
       
       
       
       
     }

     if(closest_idx != -1){

        output[idx] = sphere[closest_idx].color;
     }else{

        output[idx] =   make_uchar3(0, 0, 0);
     }
     
    
 
   
    //    printf("%f %f %f %f \n", scene->s.pos.x,scene->s.pos.y,scene->s.pos.z,scene->s.radius);
    
}

// array of spheres of size n
// add random color to each sphere
// random placement of sphere with radius

void init_sphere(Sphere *s){

s->pos.x = 3*distrib(gen);
s->pos.y = 3*distrib(gen);
s->pos.z = distrib(gen) + 4.0f;

s->radius = 1.0f;

s->color.x = distrib_color(gen_color);
s->color.y = distrib_color(gen_color);
s->color.z = distrib_color(gen_color);
}

int main()
{
 

    std::ofstream outputFile("out.ppm");

    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open or create the file!" << std::endl;
        return 1;
    }

    outputFile << "P3" << "\n";

    outputFile << WIDTH << " " << HEIGHT << "\n";

    outputFile << MAX_COLOR << '\n';

    Sphere spheres[NUM_SPHERES];

    for(int i = 0;i <NUM_SPHERES;i++){
     init_sphere(&spheres[i]);

    }
    // sphere.pos = {0.0f, 0.0f, 4.0f};
    // sphere.radius = 2.0f;
    
    
    // Scene s1;

    // s1.s = sphere;

    // Scene *s1_d;

     Sphere *spheres_d;

    cudaMalloc((void **)&spheres_d, NUM_SPHERES*sizeof(Sphere));

    cudaMemcpy(spheres_d, &spheres, NUM_SPHERES*sizeof(Sphere), cudaMemcpyHostToDevice);

    uchar3 *output_buffer_d;

    cudaMalloc((void **)&output_buffer_d, sizeof(output_buffer));

    dim3 block(16, 16);

    dim3 grid((WIDTH + (block.x - 1)) / block.x, (HEIGHT + (block.y - 1)) / block.y);

    raytrace<<<grid, block>>>(spheres_d, output_buffer_d);

    cudaDeviceSynchronize();

    cudaMemcpy(output_buffer, output_buffer_d, sizeof(output_buffer), cudaMemcpyDeviceToHost);



    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {

            outputFile << std::to_string(output_buffer[i * WIDTH + j].x);
            outputFile << " ";
            outputFile << std::to_string(output_buffer[i * WIDTH + j].y);
            outputFile << " ";
            outputFile << std::to_string(output_buffer[i * WIDTH + j].z);
            outputFile << " ";
        }
        outputFile << "\n";
    }

    outputFile.close();
}

// for (int i = 0; i < HEIGHT; i++)
// {
//     for (int j = 0; j < WIDTH; j++)
//     {
//       if(((i/8) + (j/8)) % 2 == 0){
//         outputFile << "255 255 255 ";
//       }else{
//          outputFile << "0 0 0 ";

//       }

//     }
//     outputFile << '\n';
// }

    // if (((row /(HEIGHT/8) ) + (col / (HEIGHT/8) )) % 2 == 0)
    // {
    //     output[idx] = {255, 255, 255};
    // }
    // else
    // {
    //     output[idx] = {0, 0, 0};
    // }
    // // output[idx] = {255, 255, 255};



//     inline __device__ float dot_product(float3 A, float3 B){

//     return A.x*B.x + A.y*B.y + A.z*B.z;

// }



// inline __device__ float length(float3 A){

//     return sqrtf(dot_product(A,A));
// }


// inline __device__ float3 normalize(float3 A){

//  float len = length(A);

//  return {A.x/len, A.y/len, A.z/len};
// }

// inline __device__ float3 subtract(float3 A, float3 B){

//     return {A.x - B.x , A.y - B.y, A.z - B.z};
// }

// inline __device__ float3 add(float3 A, float3 B){

//     return {A.x + B.x, A.y + B.y, A.z + B.z }
// }