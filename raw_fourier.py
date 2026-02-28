import math

def fourier_1D(input_vector: list)-> list:
    """
    Takes 1D vector as input, computes and returns fourier 
    transformed vector.
    """

    transformed_vector = []
    
    N = len(input_vector)
    for k in range(N):
        s= 0j # initialized a complex number 
        for n in range(N):
            pwr_angle = -2*math.pi*k*n/N
            s += input_vector[n]* complex(math.sin(pwr_angle), math.cos(pwr_angle))

        transformed_vector.append(s)
    return transformed_vector
def main():

    print(fourier_1D([1, 7]))

if __name__ == "__main__" :
    main()