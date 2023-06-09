#include <array>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <mpi.h>

#include "Faces.h"
#include "Mugs.h"

__global__ static void init(Faces::Double6D v)
{
  const int ix = threadIdx.x;
  const int iy = threadIdx.y;
  const int iz = blockIdx.x%blockDim.x;
  const int jx = blockIdx.x/blockDim.x;
  const int jy = blockIdx.y;
  const int jz = blockIdx.z;
  v(ix,iy,iz,jx,jy,jz) = jz*10000+jy*100+jx+iz*0.01+iy*0.0001+ix*0.000001;
}

static void ainit(Array<double,6> &v)
{
  for (int jz = 0; jz < v.size(5); jz++) {
    for (int jy = 0; jy < v.size(4); jy++) {
      for (int jx = 0; jx < v.size(3); jx++) {
        for (int iz = 0; iz < v.size(2); iz++) {
          for (int iy = 0; iy < v.size(1); iy++) {
            for (int ix = 0; ix < v.size(0); ix++) {
              v(ix,iy,iz,jx,jy,jz) = jz*10000+jy*100+jx+iz*0.01+iy*0.0001+ix*0.000001;
            }
          }
        }
      }
    }
  }
}

static void compare(Faces::Double6D &ud, const double *const __restrict v, const long nu[6], const long du[6])
{
  std::vector<double> u;
  ud.copy(u);
  std::array<int,6> ml{0,0,0,0,0,0};
  double mv = 0;
  for (int jz = 0; jz < nu[5]; jz++) {
    for (int jy = 0; jy < nu[4]; jy++) {
      for (int jx = 0; jx < nu[3]; jx++) {
        for (int iz = 0; iz < nu[2]; iz++) {
          for (int iy = 0; iy < nu[1]; iy++) {
            for (int ix = 0; ix < nu[0]; ix++) {
              const long i = ix+iy*du[0]+iz*du[1]+jx*du[2]+jy*du[3]+jz*du[4];
              const double ue = u[ud.index(ix,iy,iz,jx,jy,jz)];
              const double ve = v[i];
              const double denom = std::abs(ue)+std::abs(ve);
              const double d = (denom > 1e-15) ? std::abs(ue-ve)/denom : 0;
              if (mv < d) {
                mv = d;
                ml = {ix,iy,iz,jx,jy,jz};
              }
            }
          }
        }
      }
    }
  }

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int pass = 0;
  if (mv < 1e-15) {
    pass = 1;
    //std::cout<<rank<<" PASS"<<std::endl;
  } else {
    std::cout<<rank<<" FAIL "<<mv<<" ("<<ml[0];
    for (int i = 1; i < 6; i++) std::cout<<','<<ml[i];
    const long i = ml[0]+ml[1]*du[0]+ml[2]*du[1]+ml[3]*du[2]+ml[4]*du[3]+ml[5]*du[4];
    const double ue = u[i];
    const double ve = v[i];
    std::cout<<") "<<ue<<' '<<ve<<' '<<std::abs(ue-ve)<<' '<<std::abs(ue-ve)/(std::abs(ue)+std::abs(ve))<<std::endl;
#if 0
    for (int jz = 0; jz < nu[5]; jz++) {
      for (int jy = 0; jy < nu[4]; jy++) {
        for (int jx = 0; jx < nu[3]; jx++) {
          for (int iz = 0; iz < nu[2]; iz++) {
            for (int iy = 0; iy < nu[1]; iy++) {
              for (int ix = 0; ix < nu[0]; ix++) {
                const long i = ix+iy*du[0]+iz*du[1]+jx*du[2]+jy*du[3]+jz*du[4];
                const double ue = u[i];
                const double ve = v[i];
                const double denom = std::abs(ue)+std::abs(ve);
                const double d = denom ? std::abs(ue-ve)/denom : 0;
                if (d >= 1e-15) std::cout<<'('<<ix<<','<<iy<<','<<iz<<")@("<<jx<<','<<jy<<','<<jz<<") "<<ue<<' '<<ve<<' '<<d<<std::endl;
              }
            }
          }
        }
      }
    }
#endif
  }
  int npass = 0;
  MPI_Reduce(&pass,&npass,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  if (rank == 0) std::cout<<npass<<" tasks passed correctness check"<<std::endl;
}


int main(int argc, char *argv[])
{
  MPI_Init(&argc,&argv);

  int size = -1;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
    int lrank = MPI_PROC_NULL;
    MPI_Comm_rank(local,&lrank);
    int nd = 0;
    CHECK(gpuGetDeviceCount(&nd));
    const int target = lrank%nd;
    CHECK(gpuSetDevice(target));
    int myd = -1;
    CHECK(gpuGetDevice(&myd));
    for (int i = 0; i < size; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if ((nd > 1) && (rank == i)) std::cout<<rank<<" with node rank "<<lrank<<" using device "<<myd<<" ("<<nd<<" devices per node) (asked for "<<target<<")"<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  int lx,ly,lz,mx,my,mz,n,niface,niel,nshare;

  if (rank == 0) {
    std::cin>>lx>>ly>>lz>>mx>>my>>mz>>n>>niface>>niel>>nshare;
    std::cout<<lx<<' '<<ly<<' '<<lz<<" tasks\n";
    std::cout<<mx<<' '<<my<<' '<<mz<<" local elements of size "<<n<<'\n';
    std::cout<<niface<<" face inits x "<<niel<<" element inits x ";
    if (nshare < 0) std::cout<<-nshare<<" shares without compute kernels"<<std::endl;
    else std::cout<<nshare<<" shares"<<std::endl;
  }

  MPI_Bcast(&lx,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&ly,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&lz,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&mx,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&my,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&mz,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&niface,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&niel,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&nshare,1,MPI_INT,0,MPI_COMM_WORLD);

  bool compute = true;
  if (nshare < 0) {
    compute = false;
    nshare = -nshare;
  }

  {
    Array<double,6> v(n,n,n,mx,my,mz);
    ainit(v);
    Mugs mugs(rank,lx,ly,lz,mx,my,mz,n);
    for (int i = 0; i < nshare; i++) mugs.share(v,compute);

    Faces::Double6D u(n,n,n,mx,my,mz);
    double t = 0;
    {
      MPI_Barrier(MPI_COMM_WORLD);
      for (int k = 0; k < niface; k++) {
        if (rank == 0) std::cout<<(k+1)<<": ";
        Faces faces(rank,lx,ly,lz,mx,my,mz,n);
        for (int j = 0; j <= niel; j++) {
          init<<<dim3(mx*n,my,mz),dim3(n,n)>>>(u);
          CHECK(gpuDeviceSynchronize());
          const double tstart = MPI_Wtime();
          for (int i = 0; i < nshare; i++) faces.share(u,compute);
          CHECK(gpuDeviceSynchronize());
          const double tstop = MPI_Wtime();
          if (j > 0) t += tstop-tstart;
        }
      }
    }
    compare(u,v.data(),v.sizes(),v.strides());
    double tmax = 0;
    double tmin = 0;
    double tsum = 0;
    MPI_Reduce(&t,&tmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&t,&tmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&t,&tsum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if (rank == 0) std::cout<<"time "<<tsum/double(size)<<" avg "<<tmin<<" min "<<tmax<<" max"<<std::endl;
  }

  MPI_Finalize();
  return 0;
}
