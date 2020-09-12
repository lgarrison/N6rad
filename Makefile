TARGETS := n6rad
NVCC := nvcc

n6rad: n6rad.cu
	$(NVCC) -O3 -g -arch=sm_75 -Xcompiler "-fopenmp" --ptxas-options=-v -o $@ $<

clean:
	$(RM) $(TARGETS)
distclean: clean

.PHONY: clean distclean
