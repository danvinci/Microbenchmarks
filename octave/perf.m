%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Main function. All the tests are run here.           %%
%%  The functions declarations can be found at the end.  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function perf()

	warning off;

	f = fib(20);
	assert(f == 6765)
	timeit('recursion_fibonacci', @fib, 20)

	timeit('parse_integers', @parseintperf, 1000)

	%% array constructors %%

	%o = ones(200,200);
	%assert(all(o) == 1)
	%timeit('ones', @ones, 200, 200)

	%assert(all(matmul(o) == 200))
	%timeit('AtA', @matmul, o)

	mandel(complex(-.53,.68));
	assert(sum(sum(mandelperf(true))) == 14791)
	timeit('userfunc_mandelbrot', @mandelperf, true)

	assert(issorted(sortperf(5000)))
	timeit('recursion_quicksort', @sortperf, 5000)

	s = pisum(true);
	assert(abs(s-1.644834071848065) < 1e-12);
	timeit('iteration_pi_sum',@pisum, true)

	%s = pisumvec(true);
	%assert(abs(s-1.644834071848065) < 1e-12);
	%timeit('pi_sum_vec',@pisumvec, true)

	[s1, s2] = randmatstat(1000);
	assert(round(10*s1) > 5 && round(10*s1) < 10);
	timeit('matrix_statistics', @randmatstat, 1000)

	timeit('matrix_multiply', @randmatmul, 1000);

	printfd(1)
    timeit('print_to_file', @printfd, 100000)

    nb_ref = nbody_perf(1000, 10, 0.01);
    assert(abs(nbody_perf(1000, 10, 0.01) - nb_ref) < 1e-6);
    timeit('simulation_nbody', @nbody_perf, 1000, 10, 0.01)

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Functions declarations  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function assert(bool)
   if ~bool
     error('Assertion failed')
   end
end

function timeit(name, func, varargin)
    lang = 'matlab';
    if exist('OCTAVE_VERSION') ~= 0
       lang = 'octave';
    end

    nexpt = 5;
    times = zeros(nexpt, 1);

    for i=1:nexpt
        tic(); func(varargin{:}); times(i) = toc();
    end

    times = sort(times);
    fprintf ('%s,%s,%.8f\n', lang, name, times(1)*1000);
end

%% recursive fib %%

function f = fib(n)
    if n < 2
        f = n;
        return
    else
        f = fib(n-1) + fib(n-2);
    end
end

%% parse int %%

function n = parseintperf(t)
    for i = 1:t
        n = fix(rand*(2^32));
        s = sprintf('%08x',n);
        m = sscanf(s,'%x');
        assert(m == n);
    end
end

%% matmul and transpose %%

%function oo = matmul(o)
%    oo = o * o.';
%end

%% mandelbrot set: complex arithmetic and comprehensions %%

function r = abs2(z)
    r = real(z)*real(z) + imag(z)*imag(z);
end

function n = mandel(z)
    n = 0;
    c = z;
    for n=0:79
        if abs2(z)>4
            return
        end
        z = z^2+c;
    end
    n = 80;
end

function M = mandelperf(ignore)
    x=-2.0:.1:0.5;
    y=-1:.1:1;
    M=zeros(length(y),length(x));
    for r=1:size(M,1)
        for c=1:size(M,2)
           M(r,c) = mandel(x(c)+y(r)*i);
        end
    end
end

%% numeric vector quicksort %%

function b = qsort(a)
    b = qsort_kernel(a, 1, length(a));
end

function a = qsort_kernel(a, lo, hi)
    i = lo;
    j = hi;
    while i < hi
        pivot = a(floor((lo+hi)/2));
    	while i <= j
              while a(i) < pivot, i = i + 1; end
              while a(j) > pivot, j = j - 1; end
              if i <= j
	      	 t = a(i);
	    	 a(i) = a(j);
	    	 a(j) = t;
            	 i = i + 1;
            	 j = j - 1;
       	      end
    	end
        if lo < j; a=qsort_kernel(a, lo, j); end
        lo = i;
	    j = hi;
    end
end

function v = sortperf(n)
    v = rand(n,1);
    v = qsort(v);
end

%% slow pi series %%

function sum = pisum(ignore)
    sum = 0.0;
    for j=1:500
        sum = 0.0;
        for k=1:10000
            sum = sum + 1.0/(k*k);
        end
    end
end

%% slow pi series, vectorized %%

function s = pisumvec(ignore)
    a = [1:10000];
    for j=1:500
        s = sum( 1./(a.^2));
    end
end

%% random matrix statistics %%

function [s1, s2] = randmatstat(t)
    n=5;
    v = zeros(t,1);
    w = zeros(t,1);
    for i=1:t
        a = randn(n, n);
        b = randn(n, n);
        c = randn(n, n);
        d = randn(n, n);
        P = [a b c d];
        Q = [a b;c d];
        v(i) = trace((P.'*P)^4);
        w(i) = trace((Q.'*Q)^4);
    end
    s1 = std(v)/mean(v);
    s2 = std(w)/mean(w);
end

function t = mytranspose(x)
    [m, n] = size(x);
    t = zeros(n, m);
    for i=1:n
        for j=1:m
            t(i,j) = x(j,i);
        end
    end
end

%% largish random number gen & matmul %%

function X = randmatmul(n)
    X = rand(n,n)*rand(n,n);
end

%% printf %%

function printfd(n)
    f = fopen('/dev/null','w');
    for i = 1:n
        fprintf(f, '%d %d\n', i, i + 1);
    end
    fclose(f);
end

%% N-body simulation %%

function sys = nbody_init(n)
    rand("state", 42);
    inv_n = 1.0 / n; sys = cell(1,7);
    sys{1}=sys{2}=sys{3}=sys{4}=sys{5}=sys{6}=sys{7}=zeros(1,n);
    for i=1:n; sys{1}(i)=rand()*2-1;sys{2}(i)=rand()*2-1;sys{3}(i)=rand()*2-1;sys{4}(i)=(rand()-0.5)*0.1;sys{5}(i)=(rand()-0.5)*0.1;sys{6}(i)=(rand()-0.5)*0.1;sys{7}(i)=rand()*inv_n; end
end

function sys = nbody_step(sys, dt)
    G=1;eps2=1e-4;n=length(sys{1});
    for i=1:n; fx=fy=fz=0;
        for j=1:n; dx=sys{1}(j)-sys{1}(i);dy=sys{2}(j)-sys{2}(i);dz=sys{3}(j)-sys{3}(i);dsq=dx*dx+dy*dy+dz*dz+eps2;inv=1/sqrt(dsq);inv3=inv^3;fx=fx+dx*inv3*sys{7}(j);fy=fy+dy*inv3*sys{7}(j);fz=fz+dz*inv3*sys{7}(j); end
        sys{4}(i)=sys{4}(i)+dt*G*fx;sys{5}(i)=sys{5}(i)+dt*G*fy;sys{6}(i)=sys{6}(i)+dt*G*fz; end
    for i=1:n; sys{1}(i)=sys{1}(i)+dt*sys{4}(i);sys{2}(i)=sys{2}(i)+dt*sys{5}(i);sys{3}(i)=sys{3}(i)+dt*sys{6}(i); end
end

function cs = nbody_perf(n, steps, dt)
    sys = nbody_init(n);
    for s = 1:steps; sys = nbody_step(sys, dt); end
    cs = sum(sys{1}) + sum(sys{2}) + sum(sys{3});
end
