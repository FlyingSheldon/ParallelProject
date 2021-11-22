# Image Sharpening with Halide and Cuda 

Team member: Jiahua Huang (jiahuah), Xinna Guo (xinnag)
Webpage: https://flyingsheldon.github.io/ParallelProject/

## 1 Summary
We are going to implement several image processing algorithms including sharpening and highlight/shadow adjustment using Halide and Cuda on GPU.

## 2 Background
Sharpening increases the contrast between pixels and enhances the line structure and other details about the image. Naive approach will introduce 'halo's around the edge, so we plan to use a bilateral/trilateral filter or other recent techniques. 
 
For highlight adjustment and shadow adjustment, several target areas need to be determined first in order to perform corresponding operation on it. Then the filter weight for those target areas is determined by the neighbour pixels. 
 
The speedup of these algorithms benefit from the parallelism.
## 3 Challenge
1. Decomposition and communication for sharpening
It’s a challenge to decompose the problem. We also need to figure out a better way to assign it to each thread to minimize the communication between each thread due to the dependency on neighboring pixels.

2. Pixel dependency for highlight & shadow adjustment 
In the first step of highlight & shadow adjustment, we need to determine which category this pixel belongs to. This requires the pixel distribution of the whole image. 
This statistical stage makes the program hard to parallelize because one pixel depends on all the other pixels. Also, contention will happen if every thread tries to add the result to the shared address space at the same time. 

3. Halide domain specific language
 Halide is a programming language that is specified to make image and array processing easier and faster. None of the team members have experience with Halide so we need to start learning everything from scratch by ourselves. 

## 4 Resources
- Computers: GHC machines
- Code: We will use our code from Assignment 2 as the skeleton for CUDA initialization. 
- Sharpening: 
  - [An Image Enhancement Technique Combining Sharpening and Noise Reduction](https://ieeexplore.ieee.org/iel5/19/22392/01044761.pdf?casa_token=e8vIWpGLM7IAAAAA:lCyy04GTVAMH1lb3S6U001CrO0n6M8qj5vPwGHGwocgQM2uys6NIGJaR5Cp_8BZytX5Wf-RF1w)
  - [An Efficient and Self-Adapted Approach to the Sharpening of Color Images](https://www.hindawi.com/journals/tswj/2013/105945/)
- Halide
  - [Halide Tutorials](https://halide-lang.org/tutorials/tutorial_introduction.html)   
  - [Halide Examples](https://github.com/halide/Halide/tree/master/apps)
- Highlight & shadow adjustment: 
  - [local laplacian pyramids](https://www.darktable.org/2017/11/local-laplacian-pyramids/)
  - [What is the algorithm behind Photoshop's Highlight or shadow alteration?](https://stackoverflow.com/questions/51591445/what-is-the-algorithm-behind-photoshops-highlight-or-shadow-alteration)
  - [Image shadow and highlight correction/adjustment with opencv](https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df)

## 5 Goals and deliverables
### 5.1 Plan to achieve
The basic deliverables that we want to achieve are a working CUDA implementation and a working Halide implementation of image sharpening. The comparison of CUDA and Halide will also be analysed as well. 

*During the implementation, we notice that the user interface is not friendly. So we plan to build a GUI for this project.*

### 5.2 Hope to achieve
If everything goes ahead of the plan, we plan to try to tackle the highlight/shadow adjustment, which needs quite some time to figure out how to parallelize the first step of categorizing the pixels. 

## 6 Platform Choice
For the programming language and tools, we plan to use both Halide and CUDA. One programmer will implement the algorithms on C++  and then manually parallelize it using CUDA, while the other will implement it on Halide. 
 
GPU is designed for image processing related problems and Halide is designed to make it easier to write high-performance image and array processing code on modern machines.The advantage of domain specific languages will be analyzed as well. 

## 7 Schedule

| Week | Time      | Work |
| --------------- |------------- | ----------- |
| Week 1 | 11/4 - 11/11     | Waiting for feedback from instructors. |
| Week 2 | 11/11 - 11/18 | Research on sharpening algorithms. Set up workspace and enviroment. |
| Week 3-1 | 11/18 - 11/21 | Complete a serial version of image sharpening on C++. Finish Milestone report. |
| Week 3-2 | 11/21 - 11/25 | Start working on Halide (Xinna) and CUDA (Jiahua) implementations. |
| Week 4-1 | 11/25 - 11/28 | Finish naive Halide (Xinna) and CUDA (Jiahua) implementations. |
| Week 4-2 | 11/28 - 12/2 | Try to optimize Halide (Xinna) and CUDA (Jiahua) implementations.|
| Week 5-1 | 12/2 - 12/5 | Finish optimizaing Halide (Xinna) and CUDA (Jiahua) implementations. |
| Week 5-2 | 12/5 - 12/9 | Report the performance gain and analyze the difference between C++ and domain specific language. If time permits, explore highlight/shadow adjust.  |

## Project Milestone
The project milestone report can be found [here](https://docs.google.com/document/d/1HQVhltZfXv3lvUDZ7W8OBGb-d31G9bimXVihOkDngkw/edit?usp=sharing).

