#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <mpi.h>

#include "Faces.h"


Faces::Faces(const int id, const int lx, const int ly, const int lz, const int mx, const int my, const int mz, const int n):
  lx_(lx),ly_(ly),lz_(lz),
  mx_(mx),my_(my),mz_(mz),
  n_(n),
  xfr_(n,n,my,mz,2),
  yfr_(n,n,mx,mz,2),
  zfr_(n,n,mx,my,2),
  xfs_(n,n,my,mz,2),
  yfs_(n,n,mx,mz,2),
  zfs_(n,n,mx,my,2)
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

  if (rank_ == 0) {
    std::cout<<"Initialized Faces: "<<mx<<" x "<<my<<" x "<<mz<<" elements of order "<<n-1<<" on "<<lx<<" x "<<ly<<" x "<<lz<<" tasks"<<std::endl;
  }
}

int Faces::neighbor(const int dx, const int dy, const int dz) const
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

template <typename F>
static __global__ void run(F f)
{
  f();
}

void Faces::share(Double6D &u, const bool compute)
{
  constexpr int tag = 3;
  const int n = n_;
  const int mx = mx_;
  const int my = my_;
  const int mz = mz_;
  const int nm1 = n_-1;
  const int mxm1 = mx_-1;
  const int mym1 = my_-1;
  const int mzm1 = mz_-1;

  Double5D &xfr = xfr_;
  Double5D &yfr = yfr_;
  Double5D &zfr = zfr_;

  Double5D &xfs = xfs_;
  Double5D &yfs = yfs_;
  Double5D &zfs = zfs_;

  // post recvs in use order

  MPI_Irecv(zfr.data(0,0,0,0,0),nface_[2],MPI_DOUBLE,iface_[4],tag,MPI_COMM_WORLD,reqr_+0);
  MPI_Irecv(zfr.data(0,0,0,0,1),nface_[2],MPI_DOUBLE,iface_[5],tag,MPI_COMM_WORLD,reqr_+1);

  MPI_Irecv(yfr.data(0,0,0,0,0),nface_[1],MPI_DOUBLE,iface_[2],tag,MPI_COMM_WORLD,reqr_+2);
  MPI_Irecv(yfr.data(0,0,0,0,1),nface_[1],MPI_DOUBLE,iface_[3],tag,MPI_COMM_WORLD,reqr_+3);

  MPI_Irecv(xfr.data(0,0,0,0,0),nface_[0],MPI_DOUBLE,iface_[0],tag,MPI_COMM_WORLD,reqr_+4);
  MPI_Irecv(xfr.data(0,0,0,0,1),nface_[0],MPI_DOUBLE,iface_[1],tag,MPI_COMM_WORLD,reqr_+5);


  gpuFor({n},{n},{mx},{my},GPU_LAMBDA(const int ix, const int iy, const int jx, const int jy) {
    zfs(ix,iy,jx,jy,0) = u(ix,iy,0,jx,jy,0);
    zfs(ix,iy,jx,jy,1) = u(ix,iy,nm1,jx,jy,mzm1);
  });

  CHECK(gpuDeviceSynchronize());

  MPI_Isend(zfs.data(0,0,0,0,0),nface_[2],MPI_DOUBLE,iface_[4],tag,MPI_COMM_WORLD,reqs_+0);
  MPI_Isend(zfs.data(0,0,0,0,1),nface_[2],MPI_DOUBLE,iface_[5],tag,MPI_COMM_WORLD,reqs_+1);

#ifdef FUSE_Y

  gpuFor({n},{n},{mx},{my},{mz},GPU_LAMBDA(const int ix, const int ia, const int jx, const int jy, const int jz) {
    if (jz < mzm1) {
      const int iy = ia;
      u(ix,iy,nm1,jx,jy,jz) += u(ix,iy,0,jx,jy,jz+1);
      u(ix,iy,0,jx,jy,jz+1) = u(ix,iy,nm1,jx,jy,jz);
    }
    if (jy == 0) {
      const int iz = nm1-ia;
      if ((jz == 0) || (iz > 0)) yfs(ix,iz,jx,jz,0) = u(ix,0,iz,jx,0,jz);
      if ((jz < mzm1) && (iz == nm1)) yfs(ix,0,jx,jz+1,0) = u(ix,0,0,jx,0,jz+1);
    } else if (jy == mym1) {
      const int iz = ia;
      if ((jz == 0) || (iz > 0)) yfs(ix,iz,jx,jz,1) = u(ix,nm1,iz,jx,mym1,jz);
      if ((jz < mzm1) && (iz == nm1)) yfs(ix,0,jx,jz+1,1) = u(ix,nm1,0,jx,mym1,jz+1);
    }
  });

#else

  gpuFor({n},{n},{mx},{my},{mzm1},GPU_LAMBDA(const int ix, const int iy, const int jx, const int jy, const int jz) {
    u(ix,iy,nm1,jx,jy,jz) += u(ix,iy,0,jx,jy,jz+1);
    u(ix,iy,0,jx,jy,jz+1) = u(ix,iy,nm1,jx,jy,jz);
  });

  gpuFor({n},{n},{mx},{mz},GPU_LAMBDA(const int ix, const int iz, const int jx, const int jz) {
    yfs(ix,iz,jx,jz,0) = u(ix,0,iz,jx,0,jz);
    yfs(ix,iz,jx,jz,1) = u(ix,nm1,iz,jx,mym1,jz);
  });

#endif

  MPI_Waitall(2,reqr_+0,MPI_STATUSES_IGNORE);

  gpuFor({n},{mx},GPU_LAMBDA(const int ix, const int jx) {
    yfs(ix,0,jx,0,0) += zfr(ix,0,jx,0,0);
    yfs(ix,nm1,jx,mzm1,0) += zfr(ix,0,jx,0,1);
    yfs(ix,0,jx,0,1) += zfr(ix,nm1,jx,mym1,0);
    yfs(ix,nm1,jx,mzm1,1) += zfr(ix,nm1,jx,mym1,1);
  });

  CHECK(gpuDeviceSynchronize());

  MPI_Isend(yfs.data(0,0,0,0,0),nface_[1],MPI_DOUBLE,iface_[2],tag,MPI_COMM_WORLD,reqs_+2);
  MPI_Isend(yfs.data(0,0,0,0,1),nface_[1],MPI_DOUBLE,iface_[3],tag,MPI_COMM_WORLD,reqs_+3);

#ifdef FUSE_Z

  gpuFor({n},{n},{mx},{my},{mz},GPU_LAMBDA(const int ia, const int ib, const int jx, const int jy, const int jz) {

    if ((jz == 0) && ((jy == 0) || (ib < nm1))) u(ia,nm1-ib,0,jx,jy,0) += zfr(ia,nm1-ib,jx,jy,0);
    else if ((jz == mzm1) && ((jy == 0) || (ib > 0))) u(ia,ib,nm1,jx,jy,mzm1) += zfr(ia,ib,jx,jy,1);

    if (jy < mym1) {

      if ((jz == 0) && (ib == 0)) u(ia,0,0,jx,jy+1,0) += zfr(ia,0,jx,jy+1,0);
      else if ((jz == mzm1) && (ib == nm1)) u(ia,0,nm1,jx,jy+1,mzm1) += zfr(ia,0,jx,jy+1,1);

      u(ia,nm1,ib,jx,jy,jz) += u(ia,0,ib,jx,jy+1,jz);
      u(ia,0,ib,jx,jy+1,jz) = u(ia,nm1,ib,jx,jy,jz);
    }

    if ((jz == mzm1) && (ia == nm1)) {

      if ((jy == 0) || (ib > 0)) {
        if (jx == 0) xfs(ib,nm1,jy,jz,0) = u(0,ib,nm1,0,jy,jz);
        else if (jx == mxm1) xfs(ib,nm1,jy,jz,1) = u(nm1,ib,nm1,mxm1,jy,jz);
      }

      if ((jy < mym1) && (ib == nm1)) {
        if (jx == 0) xfs(0,nm1,jy+1,jz,0) = u(0,0,nm1,0,jy+1,jz);
        else if (jx == mxm1) xfs(0,nm1,jy+1,jz,1) = u(nm1,0,nm1,mxm1,jy+1,jz);
      }

    } else if ((jz == 0) && (ia == 0)) {

      if ((jy == 0) || (ib < nm1)) {
        if (jx == 0) xfs(nm1-ib,0,jy,jz,0) = u(0,nm1-ib,0,0,jy,jz);
        else if (jx == mxm1) xfs(nm1-ib,0,jy,jz,1) = u(nm1,nm1-ib,0,mxm1,jy,jz);
      }

      if ((jy < mym1) && (ib == 0)) {
        if (jx == 0) xfs(0,0,jy+1,jz,0) = u(0,0,0,0,jy+1,jz);
        else if (jx == mxm1) xfs(0,0,jy+1,jz,1) = u(nm1,0,0,mxm1,jy+1,jz);
      }
    }

    if (((jz > 0) || (ib > 0)) && ((jz < mzm1) || (ib < nm1))) {

      if (jx == 0) {

        if ((jy == 0) || (ia < nm1)) xfs(nm1-ia,ib,jy,jz,0) = u(0,nm1-ia,ib,0,jy,jz);
        if ((jy < mym1) && (ia == 0)) xfs(0,ib,jy+1,jz,0) = u(0,0,ib,0,jy+1,jz);

      } else if (jx == mxm1) {

        if ((jy == 0) || (ia > 0)) xfs(ia,ib,jy,jz,1) = u(nm1,ia,ib,mxm1,jy,jz);
        if ((jy < mym1) && (ia == nm1)) xfs(0,ib,jy+1,jz,1) = u(nm1,0,ib,mxm1,jy+1,jz);
      }
    }
  });

#else

  gpuFor({n},{n},{mx},{my},GPU_LAMBDA(const int ix, const int iy, const int jx, const int jy) {
    u(ix,iy,0,jx,jy,0) += zfr(ix,iy,jx,jy,0);
    u(ix,iy,nm1,jx,jy,mzm1) += zfr(ix,iy,jx,jy,1);
  });

  gpuFor({n},{n},{mx},{mym1},{mz},GPU_LAMBDA(const int ix, const int iz, const int jx, const int jy, const int jz) {
    u(ix,nm1,iz,jx,jy,jz) += u(ix,0,iz,jx,jy+1,jz);
    u(ix,0,iz,jx,jy+1,jz) = u(ix,nm1,iz,jx,jy,jz);
  });

  gpuFor({n},{n},{my},{mz},GPU_LAMBDA(const int iy, const int iz, const int jy, const int jz) {
    xfs(iy,iz,jy,jz,0) = u(0,iy,iz,0,jy,jz);
    xfs(iy,iz,jy,jz,1) = u(nm1,iy,iz,mxm1,jy,jz);
  });

#endif

  MPI_Waitall(2,reqr_+2,MPI_STATUSES_IGNORE);

  gpuFor({n},{mz},GPU_LAMBDA(const int iz, const int jz) {
    xfs(0,iz,0,jz,0) += yfr(0,iz,0,jz,0);
    xfs(nm1,iz,mym1,jz,0) += yfr(0,iz,0,jz,1);
    xfs(0,iz,0,jz,1) += yfr(nm1,iz,mxm1,jz,0);
    xfs(nm1,iz,mym1,jz,1) += yfr(nm1,iz,mxm1,jz,1);
  });

  CHECK(gpuDeviceSynchronize());

  MPI_Isend(xfs.data(0,0,0,0,0),nface_[0],MPI_DOUBLE,iface_[0],tag,MPI_COMM_WORLD,reqs_+4);
  MPI_Isend(xfs.data(0,0,0,0,1),nface_[0],MPI_DOUBLE,iface_[1],tag,MPI_COMM_WORLD,reqs_+5);

#ifdef FUSE_X

  gpuFor({n},{n},{mx},{my},{mz},GPU_LAMBDA(const int ia, const int iz, const int jx, const int jy, const int jz) {
    if (jy == 0) {
      const int ix = ia;
      if ((jx == 0) || (ix > 0)) u(ix,0,iz,jx,0,jz) += yfr(ix,iz,jx,jz,0);
      if ((jx < mxm1) && (ix == nm1)) u(0,0,iz,jx+1,0,jz) += yfr(0,iz,jx+1,jz,0);
    }
    else if (jy == mym1) {
      const int ix = nm1-ia;
      if ((jx == 0) || (ix > 0)) u(ix,nm1,iz,jx,mym1,jz) += yfr(ix,iz,jx,jz,1);
      if ((jx < mxm1) && (ix == nm1)) u(0,nm1,iz,jx+1,mym1,jz) += yfr(0,iz,jx+1,jz,1);
    }
    if (jx < mxm1) {
      const int iy = nm1-ia;
      u(nm1,iy,iz,jx,jy,jz) += u(0,iy,iz,jx+1,jy,jz);
      u(0,iy,iz,jx+1,jy,jz) = u(nm1,iy,iz,jx,jy,jz);
    }
  });

#else

  gpuFor({n},{n},{mx},{mz},GPU_LAMBDA(const int ix, const int iz, const int jx, const int jz) {
    u(ix,0,iz,jx,0,jz) += yfr(ix,iz,jx,jz,0);
    u(ix,nm1,iz,jx,mym1,jz) += yfr(ix,iz,jx,jz,1);
  });

  gpuFor({n},{n},{mxm1},{my},{mz},GPU_LAMBDA(const int iy, const int iz, const int jx, const int jy, const int jz) {
    u(nm1,iy,iz,jx,jy,jz) += u(0,iy,iz,jx+1,jy,jz);
    u(0,iy,iz,jx+1,jy,jz) = u(nm1,iy,iz,jx,jy,jz);
  });

#endif

  MPI_Waitall(2,reqr_+4,MPI_STATUSES_IGNORE);

  gpuFor({n},{n},{my},{mz},GPU_LAMBDA(const int iy, const int iz, const int jy, const int jz) {
    u(0,iy,iz,0,jy,jz) += xfr(iy,iz,jy,jz,0);
    u(nm1,iy,iz,mxm1,jy,jz) += xfr(iy,iz,jy,jz,1);
  });

  MPI_Waitall(6,reqs_,MPI_STATUSES_IGNORE);
  CHECK(gpuStreamSynchronize(0));
}
