#include <cassert>
#include <iostream>
#include <mpi.h>
#include <omp.h>

#include "Mugs.h"


Mugs::Mugs(const int id, const int lx, const int ly, const int lz, const int mx, const int my, const int mz, const int n):
  lx_(lx),ly_(ly),lz_(lz),
  mx_(mx),my_(my),mz_(mz),
  n_(n),
  rcorner_{},scorner_{},
  rxedge_(n,mx,4),ryedge_(n,my,4),rzedge_(n,mz,4),
  sxedge_(n,mx,4),syedge_(n,my,4),szedge_(n,mz,4),
  rxface_(n,n,my,mz,2),ryface_(n,n,mx,mz,2),rzface_(n,n,mx,my,2),
  sxface_(n,n,my,mz,2),syface_(n,n,mx,mz,2),szface_(n,n,mx,my,2)
{
  MPI_Comm_rank(MPI_COMM_WORLD,&rank_);
  MPI_Comm_size(MPI_COMM_WORLD,&size_);
  assert(id == rank_);

  kz_ = rank_/(lx_*ly_);
  ky_ = (rank_-kz_*lx_*ly_)/lx_;
  kx_ = rank_-(kz_*ly_+ky_)*lx_;

  iface_[0] = neighbor(-1,0,0);
  iface_[1] = neighbor(1,0,0);
  iface_[2] = neighbor(0,-1,0);
  iface_[3] = neighbor(0,1,0);
  iface_[4] = neighbor(0,0,-1);
  iface_[5] = neighbor(0,0,1);

  const int nn = n_*n_;
  nface_[0] = nn*my_*mz_;
  nface_[1] = nn*mx_*mz_;
  nface_[2] = nn*mx_*my_;

  iedge_[0] = neighbor(0,-1,-1);
  iedge_[1] = neighbor(0,1,-1);
  iedge_[2] = neighbor(0,-1,1);
  iedge_[3] = neighbor(0,1,1);

  iedge_[4] = neighbor(-1,0,-1);
  iedge_[5] = neighbor(1,0,-1);
  iedge_[6] = neighbor(-1,0,1);
  iedge_[7] = neighbor(1,0,1);

  iedge_[8] = neighbor(-1,-1,0);
  iedge_[9] = neighbor(1,-1,0);
  iedge_[10] = neighbor(-1,1,0);
  iedge_[11] = neighbor(1,1,0);

  nedge_[0] = n_*mx_;
  nedge_[1] = n_*my_;
  nedge_[2] = n_*mz_;

  icorner_[0] = neighbor(-1,-1,-1);
  icorner_[1] = neighbor(1,-1,-1);
  icorner_[2] = neighbor(-1,1,-1);
  icorner_[3] = neighbor(1,1,-1);
  icorner_[4] = neighbor(-1,-1,1);
  icorner_[5] = neighbor(1,-1,1);
  icorner_[6] = neighbor(-1,1,1);
  icorner_[7] = neighbor(1,1,1);

  if (rank_ == 0) {
    std::cout<<"Initialized Mugs: "<<mx<<" x "<<my<<" x "<<mz<<" elements of order "<<n-1<<" on "<<lx<<" x "<<ly<<" x "<<lz<<" tasks with "<<omp_get_max_threads()<<" threads"<<std::endl;
  }
}

int Mugs::neighbor(const int dx, const int dy, const int dz) const
{
  const int kx = kx_+dx;
  if ((kx < 0) || (kx >= lx_)) return MPI_PROC_NULL;
  const int ky = ky_+dy;
  if ((ky < 0) || (ky >= ly_)) return MPI_PROC_NULL;
  const int kz = kz_+dz;
  if ((kz < 0) || (kz >= lz_)) return MPI_PROC_NULL;
  const int id = (kz*ly_+ky)*lx_+kx;
  assert((id >= 0) && (id < size_));
  return id;
}


void Mugs::share(Array<double,6> &u, const bool compute)
{
  constexpr int tag = 3;

  assert(u.size(0) == n_);
  assert(u.size(1) == n_);
  assert(u.size(2) == n_);
  assert(u.size(3) == mx_);
  assert(u.size(4) == my_);
  assert(u.size(5) == mz_);

  const int mxm1 = mx_-1;
  const int mym1 = my_-1;
  const int mzm1 = mz_-1;
  const int nm1 = n_-1;

  if (size_ > 1) {

    int h = 0;

    // post recvs for remote faces
    
    for (int i = 0; i < 2; i++) {
      MPI_Irecv(&rxface_(0,0,0,0,i),nface_[0],MPI_DOUBLE,iface_[i],tag,MPI_COMM_WORLD,rreq_+h);
      h++;
      MPI_Irecv(&ryface_(0,0,0,0,i),nface_[1],MPI_DOUBLE,iface_[i+2],tag,MPI_COMM_WORLD,rreq_+h);
      h++;
      MPI_Irecv(&rzface_(0,0,0,0,i),nface_[2],MPI_DOUBLE,iface_[i+4],tag,MPI_COMM_WORLD,rreq_+h);
      h++;
    }

    // post recvs for remote edges

    for (int i = 0; i < 4; i++) {
      MPI_Irecv(&rxedge_(0,0,i),nedge_[0],MPI_DOUBLE,iedge_[i],tag,MPI_COMM_WORLD,rreq_+h);
      h++;
      MPI_Irecv(&ryedge_(0,0,i),nedge_[1],MPI_DOUBLE,iedge_[i+4],tag,MPI_COMM_WORLD,rreq_+h);
      h++;
      MPI_Irecv(&rzedge_(0,0,i),nedge_[2],MPI_DOUBLE,iedge_[i+8],tag,MPI_COMM_WORLD,rreq_+h);
      h++;
    }

    // post recvs for remote corners

    for (int i = 0; i < 8; i++) {
      MPI_Irecv(&rcorner_[i],1,MPI_DOUBLE,icorner_[i],tag,MPI_COMM_WORLD,rreq_+h);
      h++;
    }
  }

  // compute faces

  #pragma omp parallel
  {
    #pragma omp for collapse(5)
    for (int jz = 0; jz < mz_; jz++) {
      for (int jy = 0; jy < my_; jy++) {
        for (int jx = 1; jx < mx_; jx++) {
          for (int iz = 0; iz < n_; iz++) {
            for (int iy = 0; iy < n_; iy++) {
              if (compute ||
                  ((jz == 0) && (iz == 0)) ||
                  ((jz == mzm1) && (iz == nm1)) ||
                  ((jy == 0) && (iy == 0)) ||
                  ((jy == mym1) && (iy == nm1))) {
                const long ia = u.index(0,iy,iz,jx,jy,jz);
                const long ib = u.index(nm1,iy,iz,jx-1,jy,jz);
                u[ia] += u[ib];
                u[ib] = u[ia];
              }
            }
          }
        }
      }
    }
    #pragma omp for collapse(5)
    for (int jz = 0; jz < mz_; jz++) {
      for (int jy = 1; jy < my_; jy++) {
        for (int jx = 0; jx < mx_; jx++) {
          for (int iz = 0; iz < n_; iz++) {
            for (int ix = 0; ix < n_; ix++) {
              if (compute ||
                  ((jz == 0) && (iz == 0)) ||
                  ((jz == mzm1) && (iz == nm1)) ||
                  ((jx == 0) && (ix == 0)) ||
                  ((jx == mxm1) && (ix == nm1))) {
                const long ia = u.index(ix,0,iz,jx,jy,jz);
                const long ib = u.index(ix,nm1,iz,jx,jy-1,jz);
                u[ia] += u[ib];
                u[ib] = u[ia];
              }
            }
          }
        }
      }
    }
    #pragma omp for collapse(5)
    for (int jz = 1; jz < mz_; jz++) {
      for (int jy = 0; jy < my_; jy++) {
        for (int jx = 0; jx < mx_; jx++) {
          for (int iy = 0; iy < n_; iy++) {
            for (int ix = 0; ix < n_; ix++) {
              if (compute ||
                  ((jy == 0) && (iy == 0)) ||
                  ((jy == mym1) && (iy == nm1)) ||
                  ((jx == 0) && (ix == 0)) ||
                  ((jx == mxm1) && (ix == nm1))) {
                const long ia = u.index(ix,iy,0,jx,jy,jz);
                const long ib = u.index(ix,iy,nm1,jx,jy,jz-1);
                u[ia] += u[ib];
                u[ib] = u[ia];
              }
            }
          }
        }
      }
    }
  }

  if (size_ > 1) {

    // sends
    
    #pragma omp parallel
    {
      #pragma omp for collapse(2)
      for (int jz = 0; jz < mz_; jz++) {
        for (int jy = 0; jy < my_; jy++) {
          #pragma omp simd collapse(2)
          for (int iz = 0; iz < n_; iz++) {
            for (int iy = 0; iy < n_; iy++) {
              sxface_(iy,iz,jy,jz,0) = u(0,iy,iz,0,jy,jz);
              sxface_(iy,iz,jy,jz,1) = u(nm1,iy,iz,mxm1,jy,jz);
            }
          }
        }
      }
      #pragma omp for collapse(3)
      for (int jz = 0; jz < mz_; jz++) {
        for (int jx = 0; jx < mx_; jx++) {
          for (int iz = 0; iz < n_; iz++) {
            #pragma omp simd
            for (int ix = 0; ix < n_; ix++) {
              syface_(ix,iz,jx,jz,0) = u(ix,0,iz,jx,0,jz);
              syface_(ix,iz,jx,jz,1) = u(ix,nm1,iz,jx,mym1,jz);
            }
          }
        }
      }
      #pragma omp for collapse(2)
      for (int jy = 0; jy < my_; jy++) {
        for (int jx = 0; jx < mx_; jx++) {
          #pragma omp simd collapse(2)
          for (int iy = 0; iy < n_; iy++) {
            for (int ix = 0; ix < n_; ix++) {
              szface_(ix,iy,jx,jy,0) = u(ix,iy,0,jx,jy,0);
              szface_(ix,iy,jx,jy,1) = u(ix,iy,nm1,jx,jy,mzm1);
            }
          }
        }
      }

      #pragma omp for
      for (int jx = 0; jx < mx_; jx++) {
        #pragma omp simd
        for (int ix = 0; ix < n_; ix++) {
          sxedge_(ix,jx,0) = u(ix,0,0,jx,0,0);
          sxedge_(ix,jx,1) = u(ix,nm1,0,jx,mym1,0);
          sxedge_(ix,jx,2) = u(ix,0,nm1,jx,0,mzm1);
          sxedge_(ix,jx,3) = u(ix,nm1,nm1,jx,mym1,mzm1);
        }
      }
      #pragma omp for
      for (int jy = 0; jy < my_; jy++) {
        #pragma omp simd
        for (int iy = 0; iy < n_; iy++) {
          syedge_(iy,jy,0) = u(0,iy,0,0,jy,0);
          syedge_(iy,jy,1) = u(nm1,iy,0,mxm1,jy,0);
          syedge_(iy,jy,2) = u(0,iy,nm1,0,jy,mzm1);
          syedge_(iy,jy,3) = u(nm1,iy,nm1,mxm1,jy,mzm1);
        }
      }
      #pragma omp for
      for (int jz = 0; jz < mz_; jz++) {
        #pragma omp simd
        for (int iz = 0; iz < n_; iz++) {
          szedge_(iz,jz,0) = u(0,0,iz,0,0,jz);
          szedge_(iz,jz,1) = u(nm1,0,iz,mxm1,0,jz);
          szedge_(iz,jz,2) = u(0,nm1,iz,0,mym1,jz);
          szedge_(iz,jz,3) = u(nm1,nm1,iz,mxm1,mym1,jz);
        }
      }
    }
    scorner_[0] = u(0,0,0,0,0,0);
    scorner_[1] = u(nm1,0,0,mxm1,0,0);
    scorner_[2] = u(0,nm1,0,0,mym1,0);
    scorner_[3] = u(nm1,nm1,0,mxm1,mym1,0);
    scorner_[4] = u(0,0,nm1,0,0,mzm1);
    scorner_[5] = u(nm1,0,nm1,mxm1,0,mzm1);
    scorner_[6] = u(0,nm1,nm1,0,mym1,mzm1);
    scorner_[7] = u(nm1,nm1,nm1,mxm1,mym1,mzm1);

    int h = 0;

    MPI_Isend(&sxface_(0,0,0,0,0),nface_[0],MPI_DOUBLE,iface_[0],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&sxface_(0,0,0,0,1),nface_[0],MPI_DOUBLE,iface_[1],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&syface_(0,0,0,0,0),nface_[1],MPI_DOUBLE,iface_[2],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&syface_(0,0,0,0,1),nface_[1],MPI_DOUBLE,iface_[3],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&szface_(0,0,0,0,0),nface_[2],MPI_DOUBLE,iface_[4],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&szface_(0,0,0,0,1),nface_[2],MPI_DOUBLE,iface_[5],tag,MPI_COMM_WORLD,sreq_+h);
    h++;

    MPI_Isend(&sxedge_(0,0,0),nedge_[0],MPI_DOUBLE,iedge_[0],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&sxedge_(0,0,1),nedge_[0],MPI_DOUBLE,iedge_[1],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&sxedge_(0,0,2),nedge_[0],MPI_DOUBLE,iedge_[2],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&sxedge_(0,0,3),nedge_[0],MPI_DOUBLE,iedge_[3],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&syedge_(0,0,0),nedge_[1],MPI_DOUBLE,iedge_[4],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&syedge_(0,0,1),nedge_[1],MPI_DOUBLE,iedge_[5],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&syedge_(0,0,2),nedge_[1],MPI_DOUBLE,iedge_[6],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&syedge_(0,0,3),nedge_[1],MPI_DOUBLE,iedge_[7],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&szedge_(0,0,0),nedge_[2],MPI_DOUBLE,iedge_[8],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&szedge_(0,0,1),nedge_[2],MPI_DOUBLE,iedge_[9],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&szedge_(0,0,2),nedge_[2],MPI_DOUBLE,iedge_[10],tag,MPI_COMM_WORLD,sreq_+h);
    h++;
    MPI_Isend(&szedge_(0,0,3),nedge_[2],MPI_DOUBLE,iedge_[11],tag,MPI_COMM_WORLD,sreq_+h);
    h++;

    for (int i = 0; i < 8; i++) {
      MPI_Isend(&scorner_[i],1,MPI_DOUBLE,icorner_[i],tag,MPI_COMM_WORLD,sreq_+h);
      h++;
    }

    // recvs
    
    MPI_Waitall(26,rreq_,MPI_STATUSES_IGNORE);

    #pragma omp parallel
    {
      #pragma omp for collapse(4)
      for (int jz = 0; jz < mz_; jz++) {
        for (int jy = 0; jy < my_; jy++) {
          for (int iz = 0; iz < n_; iz++) {
            for (int iy = 0; iy < n_; iy++) {
              u(0,iy,iz,0,jy,jz) += rxface_(iy,iz,jy,jz,0);
              u(nm1,iy,iz,mxm1,jy,jz) += rxface_(iy,iz,jy,jz,1);
            }
          }
        }
      }
      #pragma omp for collapse(3)
      for (int jz = 0; jz < mz_; jz++) {
        for (int jx = 0; jx < mx_; jx++) {
          for (int iz = 0; iz < n_; iz++) {
            #pragma omp simd
            for (int ix = 0; ix < n_; ix++) {
              u(ix,0,iz,jx,0,jz) += ryface_(ix,iz,jx,jz,0);
              u(ix,nm1,iz,jx,mym1,jz) += ryface_(ix,iz,jx,jz,1);
            }
          }
        }
      }
      #pragma omp for collapse(2)
      for (int jy = 0; jy < my_; jy++) {
        for (int jx = 0; jx < mx_; jx++) {
          #pragma omp simd collapse(2)
          for (int iy = 0; iy < n_; iy++) {
            for (int ix = 0; ix < n_; ix++) {
              u(ix,iy,0,jx,jy,0) += rzface_(ix,iy,jx,jy,0);
              u(ix,iy,nm1,jx,jy,mzm1) += rzface_(ix,iy,jx,jy,1);
            }
          }
        }
      }

      #pragma omp for
      for (int jx = 0; jx < mx_; jx++) {
        #pragma omp simd
        for (int ix = 0; ix < n_; ix++) {
          u(ix,0,0,jx,0,0) += rxedge_(ix,jx,0);
          u(ix,nm1,0,jx,mym1,0) += rxedge_(ix,jx,1);
          u(ix,0,nm1,jx,0,mzm1) += rxedge_(ix,jx,2);
          u(ix,nm1,nm1,jx,mym1,mzm1) += rxedge_(ix,jx,3);
        }
      }
      #pragma omp for collapse(2)
      for (int jy = 0; jy < my_; jy++) {
        for (int iy = 0; iy < n_; iy++) {
          u(0,iy,0,0,jy,0) += ryedge_(iy,jy,0);
          u(nm1,iy,0,mxm1,jy,0) += ryedge_(iy,jy,1);
          u(0,iy,nm1,0,jy,mzm1) += ryedge_(iy,jy,2);
          u(nm1,iy,nm1,mxm1,jy,mzm1) += ryedge_(iy,jy,3);
        }
      }
      #pragma omp for collapse(2)
      for (int jz = 0; jz < mz_; jz++) {
        for (int iz = 0; iz < n_; iz++) {
          u(0,0,iz,0,0,jz) += rzedge_(iz,jz,0);
          u(nm1,0,iz,mxm1,0,jz) += rzedge_(iz,jz,1);
          u(0,nm1,iz,0,mym1,jz) += rzedge_(iz,jz,2);
          u(nm1,nm1,iz,mxm1,mym1,jz) += rzedge_(iz,jz,3);
        }
      }
    }

    u(0,0,0,0,0,0) += rcorner_[0];
    u(nm1,0,0,mxm1,0,0) += rcorner_[1];
    u(0,nm1,0,0,mym1,0) += rcorner_[2];
    u(nm1,nm1,0,mxm1,mym1,0) += rcorner_[3];
    u(0,0,nm1,0,0,mzm1) += rcorner_[4];
    u(nm1,0,nm1,mxm1,0,mzm1) += rcorner_[5];
    u(0,nm1,nm1,0,mym1,mzm1) += rcorner_[6];
    u(nm1,nm1,nm1,mxm1,mym1,mzm1) += rcorner_[7];

    MPI_Waitall(26,sreq_,MPI_STATUSES_IGNORE);
  }
}
