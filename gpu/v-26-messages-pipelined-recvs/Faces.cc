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
  cornerr_(1,8),
  corners_(1,8),
  xer_(n,mx,4),
  yer_(n,my,4),
  zer_(n,mz,4),
  xes_(n,mx,4),
  yes_(n,my,4),
  zes_(n,mz,4),
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

  for (int i = 0; i < nStreams_; i++) CHECK(gpuStreamCreate(stream_+i));

  if (rank_ == 0) {
    std::cout<<"Initialized Faces: "<<mx<<" x "<<my<<" x "<<mz<<" elements of order "<<n-1<<" on "<<lx<<" x "<<ly<<" x "<<lz<<" tasks"<<std::endl;
  }
}

Faces::~Faces()
{
  for (int i = 0; i < nStreams_; i++) {
    CHECK(gpuStreamDestroy(stream_[i]));
    stream_[i] = 0;
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

  Double2D &cornerr = cornerr_;
  Double2D &corners = corners_;

  Double3D &xer = xer_;
  Double3D &yer = yer_;
  Double3D &zer = zer_;

  Double3D &xes = xes_;
  Double3D &yes = yes_;
  Double3D &zes = zes_;

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
  MPI_Irecv(xer.data(0,0,0),nedge_[0],MPI_DOUBLE,iedge_[0],tag,MPI_COMM_WORLD,reqr_+4);
  MPI_Irecv(xer.data(0,0,1),nedge_[0],MPI_DOUBLE,iedge_[1],tag,MPI_COMM_WORLD,reqr_+5);
  MPI_Irecv(xer.data(0,0,2),nedge_[0],MPI_DOUBLE,iedge_[2],tag,MPI_COMM_WORLD,reqr_+6);
  MPI_Irecv(xer.data(0,0,3),nedge_[0],MPI_DOUBLE,iedge_[3],tag,MPI_COMM_WORLD,reqr_+7);

  MPI_Irecv(xfr.data(0,0,0,0,0),nface_[0],MPI_DOUBLE,iface_[0],tag,MPI_COMM_WORLD,reqr_+8);
  MPI_Irecv(xfr.data(0,0,0,0,1),nface_[0],MPI_DOUBLE,iface_[1],tag,MPI_COMM_WORLD,reqr_+9);
  MPI_Irecv(yer.data(0,0,0),nedge_[1],MPI_DOUBLE,iedge_[4],tag,MPI_COMM_WORLD,reqr_+10);
  MPI_Irecv(yer.data(0,0,1),nedge_[1],MPI_DOUBLE,iedge_[5],tag,MPI_COMM_WORLD,reqr_+11);
  MPI_Irecv(yer.data(0,0,2),nedge_[1],MPI_DOUBLE,iedge_[6],tag,MPI_COMM_WORLD,reqr_+12);
  MPI_Irecv(yer.data(0,0,3),nedge_[1],MPI_DOUBLE,iedge_[7],tag,MPI_COMM_WORLD,reqr_+13);
  MPI_Irecv(zer.data(0,0,0),nedge_[2],MPI_DOUBLE,iedge_[8],tag,MPI_COMM_WORLD,reqr_+14);
  MPI_Irecv(zer.data(0,0,1),nedge_[2],MPI_DOUBLE,iedge_[9],tag,MPI_COMM_WORLD,reqr_+15);
  MPI_Irecv(zer.data(0,0,2),nedge_[2],MPI_DOUBLE,iedge_[10],tag,MPI_COMM_WORLD,reqr_+16);
  MPI_Irecv(zer.data(0,0,3),nedge_[2],MPI_DOUBLE,iedge_[11],tag,MPI_COMM_WORLD,reqr_+17);
  MPI_Irecv(cornerr.data(0,0),1,MPI_DOUBLE,icorner_[0],tag,MPI_COMM_WORLD,reqr_+18);
  MPI_Irecv(cornerr.data(0,1),1,MPI_DOUBLE,icorner_[1],tag,MPI_COMM_WORLD,reqr_+19);
  MPI_Irecv(cornerr.data(0,2),1,MPI_DOUBLE,icorner_[2],tag,MPI_COMM_WORLD,reqr_+20);
  MPI_Irecv(cornerr.data(0,3),1,MPI_DOUBLE,icorner_[3],tag,MPI_COMM_WORLD,reqr_+21);
  MPI_Irecv(cornerr.data(0,4),1,MPI_DOUBLE,icorner_[4],tag,MPI_COMM_WORLD,reqr_+22);
  MPI_Irecv(cornerr.data(0,5),1,MPI_DOUBLE,icorner_[5],tag,MPI_COMM_WORLD,reqr_+23);
  MPI_Irecv(cornerr.data(0,6),1,MPI_DOUBLE,icorner_[6],tag,MPI_COMM_WORLD,reqr_+24);
  MPI_Irecv(cornerr.data(0,7),1,MPI_DOUBLE,icorner_[7],tag,MPI_COMM_WORLD,reqr_+25);

  // copy send messages

#ifdef FUSE_SEND

  gpuFor({n},{n},{std::max(mx,my)},{std::max(my,mz)},GPU_LAMBDA(const int ia, const int ib, const int ja, const int jb) {
    if ((ja < mx) && (jb < my)) {
      const int ix = ia;
      const int iy = ib;
      const int jx = ja;
      const int jy = jb;
      zfs(ix,iy,jx,jy,0) = u(ix,iy,0,jx,jy,0);
      zfs(ix,iy,jx,jy,1) = u(ix,iy,nm1,jx,jy,mzm1);
      if ((iy == 0) && (jy == 0)) {
        xes(ix,jx,0) = u(ix,0,0,jx,0,0);
        xes(ix,jx,2) = u(ix,0,nm1,jx,0,mzm1);
      } else if ((iy == nm1) && (jy == mym1)) {
        xes(ix,jx,1) = u(ix,nm1,0,jx,mym1,0);
        xes(ix,jx,3) = u(ix,nm1,nm1,jx,mym1,mzm1);
      }
      if ((ix == 0) && (jx == 0)) {
        yes(iy,jy,0) = u(0,iy,0,0,jy,0);
        yes(iy,jy,2) = u(0,iy,nm1,0,jy,mzm1);
      } else if ((ix == nm1) && (jx == mxm1)) {
        yes(iy,jy,1) = u(nm1,iy,0,mxm1,jy,0);
        yes(iy,jy,3) = u(nm1,iy,nm1,mxm1,jy,mzm1);
      }
      if ((ix == 0) && (iy == 0) && (jx == 0) && (jy == 0)) {
        corners(0,0) = u(0,0,0,0,0,0);
        corners(0,4) = u(0,0,nm1,0,0,mzm1);
      }
      if ((ix == nm1) && (iy == 0) && (jx == mxm1) && (jy == 0)) {
        corners(0,1) = u(nm1,0,0,mxm1,0,0);
        corners(0,5) = u(nm1,0,nm1,mxm1,0,mzm1);
      }
      if ((ix == 0) && (iy == nm1) && (jx == 0) && (jy == mym1)) {
        corners(0,2) = u(0,nm1,0,0,mym1,0);
        corners(0,6) = u(0,nm1,nm1,0,mym1,mzm1);
      }
      if ((ix == nm1) && (iy == nm1) && (jx == mxm1) && (jy == mym1)) {
        corners(0,3) = u(nm1,nm1,0,mxm1,mym1,0);
        corners(0,7) = u(nm1,nm1,nm1,mxm1,mym1,mzm1);
      }
    }
    if ((ja < mx) && (jb < mz)) {
      const int ix = ia;
      const int iz = ib;
      const int jx = ja;
      const int jz = jb;
      yfs(ix,iz,jx,jz,0) = u(ix,0,iz,jx,0,jz);
      yfs(ix,iz,jx,jz,1) = u(ix,nm1,iz,jx,mym1,jz);
      if ((ix == 0) && (jx == 0)) {
        zes(iz,jz,0) = u(0,0,iz,0,0,jz);
        zes(iz,jz,2) = u(0,nm1,iz,0,mym1,jz);
      } else if ((ix == nm1) && (jx == mxm1)) {
        zes(iz,jz,1) = u(nm1,0,iz,mxm1,0,jz);
        zes(iz,jz,3) = u(nm1,nm1,iz,mxm1,mym1,jz);
      }
    }
    if ((ja < my) && (jb < mz)) {
      const int iy = ia;
      const int iz = ib;
      const int jy = ja;
      const int jz = jb;
      xfs(iy,iz,jy,jz,0) = u(0,iy,iz,0,jy,jz);
      xfs(iy,iz,jy,jz,1) = u(nm1,iy,iz,mxm1,jy,jz);
    }
  },stream_[0]);

#else

  gpuFor({n},{n},{mx},{my},GPU_LAMBDA(const int ix, const int iy, const int jx, const int jy) {
    zfs(ix,iy,jx,jy,0) = u(ix,iy,0,jx,jy,0);
    zfs(ix,iy,jx,jy,1) = u(ix,iy,nm1,jx,jy,mzm1);
    if ((iy == 0) && (jy == 0)) {
      xes(ix,jx,0) = u(ix,0,0,jx,0,0);
      xes(ix,jx,2) = u(ix,0,nm1,jx,0,mzm1);
    } else if ((iy == nm1) && (jy == mym1)) {
      xes(ix,jx,1) = u(ix,nm1,0,jx,mym1,0);
      xes(ix,jx,3) = u(ix,nm1,nm1,jx,mym1,mzm1);
    }
    if ((ix == 0) && (jx == 0)) {
      yes(iy,jy,0) = u(0,iy,0,0,jy,0);
      yes(iy,jy,2) = u(0,iy,nm1,0,jy,mzm1);
    } else if ((ix == nm1) && (jx == mxm1)) {
      yes(iy,jy,1) = u(nm1,iy,0,mxm1,jy,0);
      yes(iy,jy,3) = u(nm1,iy,nm1,mxm1,jy,mzm1);
    }
    if ((ix == 0) && (iy == 0) && (jx == 0) && (jy == 0)) {
      corners(0,0) = u(0,0,0,0,0,0);
      corners(0,4) = u(0,0,nm1,0,0,mzm1);
    }
    if ((ix == nm1) && (iy == 0) && (jx == mxm1) && (jy == 0)) {
      corners(0,1) = u(nm1,0,0,mxm1,0,0);
      corners(0,5) = u(nm1,0,nm1,mxm1,0,mzm1);
    }
    if ((ix == 0) && (iy == nm1) && (jx == 0) && (jy == mym1)) {
      corners(0,2) = u(0,nm1,0,0,mym1,0);
      corners(0,6) = u(0,nm1,nm1,0,mym1,mzm1);
    }
    if ((ix == nm1) && (iy == nm1) && (jx == mxm1) && (jy == mym1)) {
      corners(0,3) = u(nm1,nm1,0,mxm1,mym1,0);
      corners(0,7) = u(nm1,nm1,nm1,mxm1,mym1,mzm1);
    }
  },stream_[0]);

  gpuFor({n},{n},{mx},{mz},GPU_LAMBDA(const int ix, const int iz, const int jx, const int jz) {
    yfs(ix,iz,jx,jz,0) = u(ix,0,iz,jx,0,jz);
    yfs(ix,iz,jx,jz,1) = u(ix,nm1,iz,jx,mym1,jz);
    if ((ix == 0) && (jx == 0)) {
      zes(iz,jz,0) = u(0,0,iz,0,0,jz);
      zes(iz,jz,2) = u(0,nm1,iz,0,mym1,jz);
    } else if ((ix == nm1) && (jx == mxm1)) {
      zes(iz,jz,1) = u(nm1,0,iz,mxm1,0,jz);
      zes(iz,jz,3) = u(nm1,nm1,iz,mxm1,mym1,jz);
    }
  },stream_[0]);

  gpuFor({n},{n},{my},{mz},GPU_LAMBDA(const int iy, const int iz, const int jy, const int jz) {
    xfs(iy,iz,jy,jz,0) = u(0,iy,iz,0,jy,jz);
    xfs(iy,iz,jy,jz,1) = u(nm1,iy,iz,mxm1,jy,jz);
  },stream_[0]);

#endif

  // compute internal faces, edges, and corners
  
  if (compute) {

#ifdef FUSE_INNER

    gpuFor({1,nm1},{1,nm1},{mx},{my},{mz},GPU_LAMBDA(const int i0, const int i1, const int jx, const int jy, const int jz) {
      if (jz > 0) {
        const int ix = i0;
        const int iy = i1;
        u(ix,iy,0,jx,jy,jz) += u(ix,iy,nm1,jx,jy,jz-1);
        u(ix,iy,nm1,jx,jy,jz-1) = u(ix,iy,0,jx,jy,jz);
      }
      if (jy > 0) {
        const int ix = i0;
        const int iz = i1;
        u(ix,0,iz,jx,jy,jz) += u(ix,nm1,iz,jx,jy-1,jz);
        u(ix,nm1,iz,jx,jy-1,jz) = u(ix,0,iz,jx,jy,jz);
      }
      if (jx > 0) {
        const int iy = i0;
        const int iz = i1;
        u(0,iy,iz,jx,jy,jz) += u(nm1,iy,iz,jx-1,jy,jz);
        u(nm1,iy,iz,jx-1,jy,jz) = u(0,iy,iz,jx,jy,jz);
      }
      if (i1 == 1) {
        if ((jx > 0) && (jy > 0)) {
          const int iz = i0;
          u(0,0,iz,jx,jy,jz) += u(nm1,0,iz,jx-1,jy,jz)+u(0,nm1,iz,jx,jy-1,jz)+u(nm1,nm1,iz,jx-1,jy-1,jz);
          u(nm1,0,iz,jx-1,jy,jz) = u(0,nm1,iz,jx,jy-1,jz) = u(nm1,nm1,iz,jx-1,jy-1,jz) = u(0,0,iz,jx,jy,jz);
        }
        if ((jx > 0) && (jz > 0)) {
          const int iy = i0;
          u(0,iy,0,jx,jy,jz) += u(nm1,iy,0,jx-1,jy,jz)+u(0,iy,nm1,jx,jy,jz-1)+u(nm1,iy,nm1,jx-1,jy,jz-1);
          u(nm1,iy,0,jx-1,jy,jz) = u(0,iy,nm1,jx,jy,jz-1) = u(nm1,iy,nm1,jx-1,jy,jz-1) = u(0,iy,0,jx,jy,jz);
        }
        if ((jy > 0) && (jz > 0)) {
          const int ix = i0;
          u(ix,0,0,jx,jy,jz) += u(ix,nm1,0,jx,jy-1,jz)+u(ix,0,nm1,jx,jy,jz-1)+u(ix,nm1,nm1,jx,jy-1,jz-1);
          u(ix,nm1,0,jx,jy-1,jz) = u(ix,0,nm1,jx,jy,jz-1) = u(ix,nm1,nm1,jx,jy-1,jz-1) = u(ix,0,0,jx,jy,jz);
        }
        if ((jx > 0) && (jy > 0) && (jz > 0) && (i0 == 1)) {
          u(0,0,0,jx,jy,jz) += u(nm1,0,0,jx-1,jy,jz)+u(0,nm1,0,jx,jy-1,jz)+u(nm1,nm1,0,jx-1,jy-1,jz)+u(0,0,nm1,jx,jy,jz-1)+u(nm1,0,nm1,jx-1,jy,jz-1)+u(0,nm1,nm1,jx,jy-1,jz-1)+u(nm1,nm1,nm1,jx-1,jy-1,jz-1);
          u(nm1,0,0,jx-1,jy,jz) = u(0,nm1,0,jx,jy-1,jz) = u(nm1,nm1,0,jx-1,jy-1,jz) = u(0,0,nm1,jx,jy,jz-1) = u(nm1,0,nm1,jx-1,jy,jz-1) = u(0,nm1,nm1,jx,jy-1,jz-1) = u(nm1,nm1,nm1,jx-1,jy-1,jz-1) = u(0,0,0,jx,jy,jz);
        }
      }
    },stream_[1]);

#else

    gpuFor({1,nm1},{1,nm1},{mx},{my},{1,mz},GPU_LAMBDA(const int ix, const int iy, const int jx, const int jy, const int jz) {
      u(ix,iy,0,jx,jy,jz) += u(ix,iy,nm1,jx,jy,jz-1);
      u(ix,iy,nm1,jx,jy,jz-1) = u(ix,iy,0,jx,jy,jz);
      if ((jx > 0) && (ix == 1)) {
        u(0,iy,0,jx,jy,jz) += u(nm1,iy,0,jx-1,jy,jz)+u(0,iy,nm1,jx,jy,jz-1)+u(nm1,iy,nm1,jx-1,jy,jz-1);
        u(nm1,iy,0,jx-1,jy,jz) = u(0,iy,nm1,jx,jy,jz-1) = u(nm1,iy,nm1,jx-1,jy,jz-1) = u(0,iy,0,jx,jy,jz);
        if ((jy > 0) && (iy == 1)) {
          u(0,0,0,jx,jy,jz) += u(nm1,0,0,jx-1,jy,jz)+u(0,nm1,0,jx,jy-1,jz)+u(nm1,nm1,0,jx-1,jy-1,jz)+u(0,0,nm1,jx,jy,jz-1)+u(nm1,0,nm1,jx-1,jy,jz-1)+u(0,nm1,nm1,jx,jy-1,jz-1)+u(nm1,nm1,nm1,jx-1,jy-1,jz-1);
          u(nm1,0,0,jx-1,jy,jz) = u(0,nm1,0,jx,jy-1,jz) = u(nm1,nm1,0,jx-1,jy-1,jz) = u(0,0,nm1,jx,jy,jz-1) = u(nm1,0,nm1,jx-1,jy,jz-1) = u(0,nm1,nm1,jx,jy-1,jz-1) = u(nm1,nm1,nm1,jx-1,jy-1,jz-1) = u(0,0,0,jx,jy,jz);
        }
      }
    },stream_[1]);

    gpuFor({1,nm1},{1,nm1},{mx},{1,my},{mz},GPU_LAMBDA(const int ix, const int iz, const int jx, const int jy, const int jz) {
      u(ix,0,iz,jx,jy,jz) += u(ix,nm1,iz,jx,jy-1,jz);
      u(ix,nm1,iz,jx,jy-1,jz) = u(ix,0,iz,jx,jy,jz);
      if ((jz > 0) && (iz == 1)) {
        u(ix,0,0,jx,jy,jz) += u(ix,nm1,0,jx,jy-1,jz)+u(ix,0,nm1,jx,jy,jz-1)+u(ix,nm1,nm1,jx,jy-1,jz-1);
        u(ix,nm1,0,jx,jy-1,jz) = u(ix,0,nm1,jx,jy,jz-1) = u(ix,nm1,nm1,jx,jy-1,jz-1) = u(ix,0,0,jx,jy,jz);
      }
    },stream_[1]);

    gpuFor({1,nm1},{1,nm1},{1,mx},{my},{mz},GPU_LAMBDA(const int iy, const int iz, const int jx, const int jy, const int jz) {
      u(0,iy,iz,jx,jy,jz) += u(nm1,iy,iz,jx-1,jy,jz);
      u(nm1,iy,iz,jx-1,jy,jz) = u(0,iy,iz,jx,jy,jz);
      if ((jy > 0) && (iy == 1)) {
        u(0,0,iz,jx,jy,jz) += u(nm1,0,iz,jx-1,jy,jz)+u(0,nm1,iz,jx,jy-1,jz)+u(nm1,nm1,iz,jx-1,jy-1,jz);
        u(nm1,0,iz,jx-1,jy,jz) = u(0,nm1,iz,jx,jy-1,jz) = u(nm1,nm1,iz,jx-1,jy-1,jz) = u(0,0,iz,jx,jy,jz);
      }
    },stream_[1]);

#endif

  }

  // send in use order

  CHECK(gpuStreamSynchronize(stream_[0]));

  MPI_Isend(zfs.data(0,0,0,0,0),nface_[2],MPI_DOUBLE,iface_[4],tag,MPI_COMM_WORLD,reqs_+0);
  MPI_Isend(zfs.data(0,0,0,0,1),nface_[2],MPI_DOUBLE,iface_[5],tag,MPI_COMM_WORLD,reqs_+1);

  MPI_Isend(yfs.data(0,0,0,0,0),nface_[1],MPI_DOUBLE,iface_[2],tag,MPI_COMM_WORLD,reqs_+2);
  MPI_Isend(yfs.data(0,0,0,0,1),nface_[1],MPI_DOUBLE,iface_[3],tag,MPI_COMM_WORLD,reqs_+3);
  MPI_Isend(xes.data(0,0,0),nedge_[0],MPI_DOUBLE,iedge_[0],tag,MPI_COMM_WORLD,reqs_+4);
  MPI_Isend(xes.data(0,0,1),nedge_[0],MPI_DOUBLE,iedge_[1],tag,MPI_COMM_WORLD,reqs_+5);
  MPI_Isend(xes.data(0,0,2),nedge_[0],MPI_DOUBLE,iedge_[2],tag,MPI_COMM_WORLD,reqs_+6);
  MPI_Isend(xes.data(0,0,3),nedge_[0],MPI_DOUBLE,iedge_[3],tag,MPI_COMM_WORLD,reqs_+7);

  MPI_Isend(xfs.data(0,0,0,0,0),nface_[0],MPI_DOUBLE,iface_[0],tag,MPI_COMM_WORLD,reqs_+8);
  MPI_Isend(xfs.data(0,0,0,0,1),nface_[0],MPI_DOUBLE,iface_[1],tag,MPI_COMM_WORLD,reqs_+9);
  MPI_Isend(yes.data(0,0,0),nedge_[1],MPI_DOUBLE,iedge_[4],tag,MPI_COMM_WORLD,reqs_+10);
  MPI_Isend(yes.data(0,0,1),nedge_[1],MPI_DOUBLE,iedge_[5],tag,MPI_COMM_WORLD,reqs_+11);
  MPI_Isend(yes.data(0,0,2),nedge_[1],MPI_DOUBLE,iedge_[6],tag,MPI_COMM_WORLD,reqs_+12);
  MPI_Isend(yes.data(0,0,3),nedge_[1],MPI_DOUBLE,iedge_[7],tag,MPI_COMM_WORLD,reqs_+13);
  MPI_Isend(zes.data(0,0,0),nedge_[2],MPI_DOUBLE,iedge_[8],tag,MPI_COMM_WORLD,reqs_+14);
  MPI_Isend(zes.data(0,0,1),nedge_[2],MPI_DOUBLE,iedge_[9],tag,MPI_COMM_WORLD,reqs_+15);
  MPI_Isend(zes.data(0,0,2),nedge_[2],MPI_DOUBLE,iedge_[10],tag,MPI_COMM_WORLD,reqs_+16);
  MPI_Isend(zes.data(0,0,3),nedge_[2],MPI_DOUBLE,iedge_[11],tag,MPI_COMM_WORLD,reqs_+17);
  MPI_Isend(corners.data(0,0),1,MPI_DOUBLE,icorner_[0],tag,MPI_COMM_WORLD,reqs_+18);
  MPI_Isend(corners.data(0,1),1,MPI_DOUBLE,icorner_[1],tag,MPI_COMM_WORLD,reqs_+19);
  MPI_Isend(corners.data(0,2),1,MPI_DOUBLE,icorner_[2],tag,MPI_COMM_WORLD,reqs_+20);
  MPI_Isend(corners.data(0,3),1,MPI_DOUBLE,icorner_[3],tag,MPI_COMM_WORLD,reqs_+21);
  MPI_Isend(corners.data(0,4),1,MPI_DOUBLE,icorner_[4],tag,MPI_COMM_WORLD,reqs_+22);
  MPI_Isend(corners.data(0,5),1,MPI_DOUBLE,icorner_[5],tag,MPI_COMM_WORLD,reqs_+23);
  MPI_Isend(corners.data(0,6),1,MPI_DOUBLE,icorner_[6],tag,MPI_COMM_WORLD,reqs_+24);
  MPI_Isend(corners.data(0,7),1,MPI_DOUBLE,icorner_[7],tag,MPI_COMM_WORLD,reqs_+25);

  // compute faces, edges, and corners
 
#ifdef FUSE_RECV

  MPI_Waitall(26,reqr_,MPI_STATUSES_IGNORE);

  gpuFor({nm1},{nm1},{std::max(mx,my)},{std::max(my,mz)},GPU_LAMBDA(const int ia, const int ib, const int ja, const int jb) {
    if ((ja < mx) && (jb < my)) {
      const int ix = ia;
      const int iy = ib;
      const int jx = ja;
      const int jy = jb;
      if ((ix == 0) && (iy == 0) && (jx > 0) && (jy > 0)) {
        u(0,0,0,jx,jy,0) += u(nm1,0,0,jx-1,jy,0)+u(0,nm1,0,jx,jy-1,0)+u(nm1,nm1,0,jx-1,jy-1,0)+zfr(0,0,jx,jy,0)+zfr(nm1,0,jx-1,jy,0)+zfr(0,nm1,jx,jy-1,0)+zfr(nm1,nm1,jx-1,jy-1,0);
        u(nm1,0,0,jx-1,jy,0) = u(0,nm1,0,jx,jy-1,0) = u(nm1,nm1,0,jx-1,jy-1,0) = u(0,0,0,jx,jy,0);
        u(0,0,nm1,jx,jy,mzm1) += u(nm1,0,nm1,jx-1,jy,mzm1)+u(0,nm1,nm1,jx,jy-1,mzm1)+u(nm1,nm1,nm1,jx-1,jy-1,mzm1)+zfr(0,0,jx,jy,1)+zfr(nm1,0,jx-1,jy,1)+zfr(0,nm1,jx,jy-1,1)+zfr(nm1,nm1,jx-1,jy-1,1);
        u(nm1,0,nm1,jx-1,jy,mzm1) = u(0,nm1,nm1,jx,jy-1,mzm1) = u(nm1,nm1,nm1,jx-1,jy-1,mzm1) = u(0,0,nm1,jx,jy,mzm1);
      } else if ((ix > 0) && (iy == 0) && (jy > 0)) {
        u(ix,0,0,jx,jy,0) += u(ix,nm1,0,jx,jy-1,0)+zfr(ix,0,jx,jy,0)+zfr(ix,nm1,jx,jy-1,0);
        u(ix,nm1,0,jx,jy-1,0) = u(ix,0,0,jx,jy,0);
        u(ix,0,nm1,jx,jy,mzm1) += u(ix,nm1,nm1,jx,jy-1,mzm1)+zfr(ix,0,jx,jy,1)+zfr(ix,nm1,jx,jy-1,1);
        u(ix,nm1,nm1,jx,jy-1,mzm1) = u(ix,0,nm1,jx,jy,mzm1);
      } else if ((ix == 0) && (iy > 0) && (jx > 0)) {
        u(0,iy,0,jx,jy,0) += u(nm1,iy,0,jx-1,jy,0)+zfr(0,iy,jx,jy,0)+zfr(nm1,iy,jx-1,jy,0);
        u(nm1,iy,0,jx-1,jy,0) = u(0,iy,0,jx,jy,0);
        u(0,iy,nm1,jx,jy,mzm1) += u(nm1,iy,nm1,jx-1,jy,mzm1)+zfr(0,iy,jx,jy,1)+zfr(nm1,iy,jx-1,jy,1);
        u(nm1,iy,nm1,jx-1,jy,mzm1) = u(0,iy,nm1,jx,jy,mzm1);
      } else if ((ix > 0) && (iy > 0)) {
        u(ix,iy,0,jx,jy,0) += zfr(ix,iy,jx,jy,0);
        u(ix,iy,nm1,jx,jy,mzm1) += zfr(ix,iy,jx,jy,1);
      }
    } 
    if ((ja < mx) && (jb < mz)) {
      const int ix = ia;
      const int iz = ib;
      const int jx = ja;
      const int jz = jb;
      if ((ix == 0) && (iz == 0) && (jx > 0) && (jz == 0)) {
        u(0,0,0,jx,0,0) += u(nm1,0,0,jx-1,0,0)+xer(0,jx,0)+xer(nm1,jx-1,0)+yfr(0,0,jx,0,0)+yfr(nm1,0,jx-1,0,0)+zfr(0,0,jx,0,0)+zfr(nm1,0,jx-1,0,0);
        u(nm1,0,0,jx-1,0,0) = u(0,0,0,jx,0,0);
        u(0,nm1,0,jx,mym1,0) += u(nm1,nm1,0,jx-1,mym1,0)+xer(0,jx,1)+xer(nm1,jx-1,1)+yfr(0,0,jx,0,1)+yfr(nm1,0,jx-1,0,1)+zfr(0,nm1,jx,mym1,0)+zfr(nm1,nm1,jx-1,mym1,0);
        u(nm1,nm1,0,jx-1,mym1,0) = u(0,nm1,0,jx,mym1,0);
        u(0,0,nm1,jx,0,mzm1) += u(nm1,0,nm1,jx-1,0,mzm1)+xer(0,jx,2)+xer(nm1,jx-1,2)+yfr(0,nm1,jx,mzm1,0)+yfr(nm1,nm1,jx-1,mzm1,0)+zfr(0,0,jx,0,1)+zfr(nm1,0,jx-1,0,1);
        u(nm1,0,nm1,jx-1,0,mzm1) = u(0,0,nm1,jx,0,mzm1);
        u(0,nm1,nm1,jx,mym1,mzm1) += u(nm1,nm1,nm1,jx-1,mym1,mzm1)+xer(0,jx,3)+xer(nm1,jx-1,3)+yfr(0,nm1,jx,mzm1,1)+yfr(nm1,nm1,jx-1,mzm1,1)+zfr(0,nm1,jx,mym1,1)+zfr(nm1,nm1,jx-1,mym1,1);
        u(nm1,nm1,nm1,jx-1,mym1,mzm1) = u(0,nm1,nm1,jx,mym1,mzm1);
      } else if ((ix > 0) && (iz == 0) && (jz == 0)) {
        u(ix,0,0,jx,0,0) += xer(ix,jx,0)+yfr(ix,0,jx,0,0)+zfr(ix,0,jx,0,0);
        u(ix,nm1,0,jx,mym1,0) += xer(ix,jx,1)+yfr(ix,0,jx,0,1)+zfr(ix,nm1,jx,mym1,0);
        u(ix,0,nm1,jx,0,mzm1) += xer(ix,jx,2)+yfr(ix,nm1,jx,mzm1,0)+zfr(ix,0,jx,0,1);
        u(ix,nm1,nm1,jx,mym1,mzm1) += xer(ix,jx,3)+yfr(ix,nm1,jx,mzm1,1)+zfr(ix,nm1,jx,mym1,1);
      } else if ((ix == 0) && (iz == 0) && (jx > 0) && (jz > 0)) {
        u(0,0,0,jx,0,jz) += u(nm1,0,0,jx-1,0,jz)+u(0,0,nm1,jx,0,jz-1)+u(nm1,0,nm1,jx-1,0,jz-1)+yfr(0,0,jx,jz,0)+yfr(nm1,0,jx-1,jz,0)+yfr(0,nm1,jx,jz-1,0)+yfr(nm1,nm1,jx-1,jz-1,0);
        u(nm1,0,0,jx-1,0,jz) = u(0,0,nm1,jx,0,jz-1) = u(nm1,0,nm1,jx-1,0,jz-1) = u(0,0,0,jx,0,jz);
        u(0,nm1,0,jx,mym1,jz) += u(nm1,nm1,0,jx-1,mym1,jz)+u(0,nm1,nm1,jx,mym1,jz-1)+u(nm1,nm1,nm1,jx-1,mym1,jz-1)+yfr(0,0,jx,jz,1)+yfr(nm1,0,jx-1,jz,1)+yfr(0,nm1,jx,jz-1,1)+yfr(nm1,nm1,jx-1,jz-1,1);
        u(nm1,nm1,0,jx-1,mym1,jz) = u(0,nm1,nm1,jx,mym1,jz-1) = u(nm1,nm1,nm1,jx-1,mym1,jz-1) = u(0,nm1,0,jx,mym1,jz);
      } else if ((ix == 0) && (iz > 0) && (jx > 0)) {
        u(0,0,iz,jx,0,jz) += u(nm1,0,iz,jx-1,0,jz)+yfr(0,iz,jx,jz,0)+yfr(nm1,iz,jx-1,jz,0);
        u(nm1,0,iz,jx-1,0,jz) = u(0,0,iz,jx,0,jz);
        u(0,nm1,iz,jx,mym1,jz) += u(nm1,nm1,iz,jx-1,mym1,jz)+yfr(0,iz,jx,jz,1)+yfr(nm1,iz,jx-1,jz,1);
        u(nm1,nm1,iz,jx-1,mym1,jz) = u(0,nm1,iz,jx,mym1,jz);
      } else if ((ix > 0) && (iz == 0) && (jz > 0)) {
        u(ix,0,0,jx,0,jz) += u(ix,0,nm1,jx,0,jz-1)+yfr(ix,0,jx,jz,0)+yfr(ix,nm1,jx,jz-1,0);
        u(ix,0,nm1,jx,0,jz-1) = u(ix,0,0,jx,0,jz);
        u(ix,nm1,0,jx,mym1,jz) += u(ix,nm1,nm1,jx,mym1,jz-1)+yfr(ix,0,jx,jz,1)+yfr(ix,nm1,jx,jz-1,1);
        u(ix,nm1,nm1,jx,mym1,jz-1) = u(ix,nm1,0,jx,mym1,jz);
      } else if ((ix > 0) && (iz > 0)) {
        u(ix,0,iz,jx,0,jz) += yfr(ix,iz,jx,jz,0);
        u(ix,nm1,iz,jx,mym1,jz) += yfr(ix,iz,jx,jz,1);
      }
    }
    if ((ja < my) && (jb < mz)) {
      const int iy = ia;
      const int iz = ib;
      const int jy = ja;
      const int jz = jb;
      if ((iy == 0) && (iz == 0) && (jy == 0) && (jz == 0)) {
        u(0,0,0,0,0,0) += cornerr(0,0)+xer(0,0,0)+yer(0,0,0)+zer(0,0,0)+xfr(0,0,0,0,0)+yfr(0,0,0,0,0)+zfr(0,0,0,0,0);
        u(nm1,0,0,mxm1,0,0) += cornerr(0,1)+xer(nm1,mxm1,0)+yer(0,0,1)+zer(0,0,1)+xfr(0,0,0,0,1)+yfr(nm1,0,mxm1,0,0)+zfr(nm1,0,mxm1,0,0);
        u(0,nm1,0,0,mym1,0) += cornerr(0,2)+xer(0,0,1)+yer(nm1,mym1,0)+zer(0,0,2)+xfr(nm1,0,mym1,0,0)+yfr(0,0,0,0,1)+zfr(0,nm1,0,mym1,0);
        u(nm1,nm1,0,mxm1,mym1,0) += cornerr(0,3)+xer(nm1,mxm1,1)+yer(nm1,mym1,1)+zer(0,0,3)+xfr(nm1,0,mym1,0,1)+yfr(nm1,0,mxm1,0,1)+zfr(nm1,nm1,mxm1,mym1,0);
        u(0,0,nm1,0,0,mzm1) += cornerr(0,4)+xer(0,0,2)+yer(0,0,2)+zer(nm1,mzm1,0)+xfr(0,nm1,0,mzm1,0)+yfr(0,nm1,0,mzm1,0)+zfr(0,0,0,0,1);
        u(nm1,0,nm1,mxm1,0,mzm1) += cornerr(0,5)+xer(nm1,mxm1,2)+yer(0,0,3)+zer(nm1,mzm1,1)+xfr(0,nm1,0,mzm1,1)+yfr(nm1,nm1,mxm1,mzm1,0)+zfr(nm1,0,mxm1,0,1);
        u(0,nm1,nm1,0,mym1,mzm1) += cornerr(0,6)+xer(0,0,3)+yer(nm1,mym1,2)+zer(nm1,mzm1,2)+xfr(nm1,nm1,mym1,mzm1,0)+yfr(0,nm1,0,mzm1,1)+zfr(0,nm1,0,mym1,1);
        u(nm1,nm1,nm1,mxm1,mym1,mzm1) += cornerr(0,7)+xer(nm1,mxm1,3)+yer(nm1,mym1,3)+zer(nm1,mzm1,3)+xfr(nm1,nm1,mym1,mzm1,1)+yfr(nm1,nm1,mxm1,mzm1,1)+zfr(nm1,nm1,mxm1,mym1,1);
      } else if ((iy == 0) && (iz == 0) && (jy == 0) && (jz > 0)) {
        u(0,0,0,0,0,jz) += u(0,0,nm1,0,0,jz-1)+xfr(0,0,0,jz,0)+xfr(0,nm1,0,jz-1,0)+yfr(0,0,0,jz,0)+yfr(0,nm1,0,jz-1,0)+zer(0,jz,0)+zer(nm1,jz-1,0);
        u(0,0,nm1,0,0,jz-1) = u(0,0,0,0,0,jz);
        u(nm1,0,0,mxm1,0,jz) += u(nm1,0,nm1,mxm1,0,jz-1)+xfr(0,0,0,jz,1)+xfr(0,nm1,0,jz-1,1)+yfr(nm1,0,mxm1,jz,0)+yfr(nm1,nm1,mxm1,jz-1,0)+zer(0,jz,1)+zer(nm1,jz-1,1);
        u(nm1,0,nm1,mxm1,0,jz-1) = u(nm1,0,0,mxm1,0,jz);
        u(0,nm1,0,0,mym1,jz) += u(0,nm1,nm1,0,mym1,jz-1)+xfr(nm1,0,mym1,jz,0)+xfr(nm1,nm1,mym1,jz-1,0)+yfr(0,0,0,jz,1)+yfr(0,nm1,0,jz-1,1)+zer(0,jz,2)+zer(nm1,jz-1,2);
        u(0,nm1,nm1,0,mym1,jz-1) = u(0,nm1,0,0,mym1,jz);
        u(nm1,nm1,0,mxm1,mym1,jz) += u(nm1,nm1,nm1,mxm1,mym1,jz-1)+xfr(nm1,0,mym1,jz,1)+xfr(nm1,nm1,mym1,jz-1,1)+yfr(nm1,0,mxm1,jz,1)+yfr(nm1,nm1,mxm1,jz-1,1)+zer(0,jz,3)+zer(nm1,jz-1,3);
        u(nm1,nm1,nm1,mxm1,mym1,jz-1) = u(nm1,nm1,0,mxm1,mym1,jz);
      } else if ((iy == 0) && (iz == 0) && (jy > 0) && (jz == 0)) {
        u(0,0,0,0,jy,0) += u(0,nm1,0,0,jy-1,0)+xfr(0,0,jy,0,0)+xfr(nm1,0,jy-1,0,0)+yer(0,jy,0)+yer(nm1,jy-1,0)+zfr(0,0,0,jy,0)+zfr(0,nm1,0,jy-1,0);
        u(0,nm1,0,0,jy-1,0) = u(0,0,0,0,jy,0);
        u(nm1,0,0,mxm1,jy,0) += u(nm1,nm1,0,mxm1,jy-1,0)+xfr(0,0,jy,0,1)+xfr(nm1,0,jy-1,0,1)+yer(0,jy,1)+yer(nm1,jy-1,1)+zfr(nm1,0,mxm1,jy,0)+zfr(nm1,nm1,mxm1,jy-1,0);
        u(nm1,nm1,0,mxm1,jy-1,0) = u(nm1,0,0,mxm1,jy,0);
        u(0,0,nm1,0,jy,mzm1) += u(0,nm1,nm1,0,jy-1,mzm1)+xfr(0,nm1,jy,mzm1,0)+xfr(nm1,nm1,jy-1,mzm1,0)+yer(0,jy,2)+yer(nm1,jy-1,2)+zfr(0,0,0,jy,1)+zfr(0,nm1,0,jy-1,1);
        u(0,nm1,nm1,0,jy-1,mzm1) = u(0,0,nm1,0,jy,mzm1);
        u(nm1,0,nm1,mxm1,jy,mzm1) += u(nm1,nm1,nm1,mxm1,jy-1,mzm1)+xfr(0,nm1,jy,mzm1,1)+xfr(nm1,nm1,jy-1,mzm1,1)+yer(0,jy,3)+yer(nm1,jy-1,3)+zfr(nm1,0,mxm1,jy,1)+zfr(nm1,nm1,mxm1,jy-1,1);
        u(nm1,nm1,nm1,mxm1,jy-1,mzm1) = u(nm1,0,nm1,mxm1,jy,mzm1);
      } else if ((iy == 0) && (iz > 0) && (jy == 0)) {
        u(0,0,iz,0,0,jz) += xfr(0,iz,0,jz,0)+yfr(0,iz,0,jz,0)+zer(iz,jz,0);
        u(nm1,0,iz,mxm1,0,jz) += xfr(0,iz,0,jz,1)+yfr(nm1,iz,mxm1,jz,0)+zer(iz,jz,1);
        u(0,nm1,iz,0,mym1,jz) += xfr(nm1,iz,mym1,jz,0)+yfr(0,iz,0,jz,1)+zer(iz,jz,2);
        u(nm1,nm1,iz,mxm1,mym1,jz) += xfr(nm1,iz,mym1,jz,1)+yfr(nm1,iz,mxm1,jz,1)+zer(iz,jz,3);
      } else if ((iy > 0) && (iz == 0) && (jz == 0)) { 
        u(0,iy,0,0,jy,0) += xfr(iy,0,jy,0,0)+yer(iy,jy,0)+zfr(0,iy,0,jy,0);
        u(nm1,iy,0,mxm1,jy,0) += xfr(iy,0,jy,0,1)+yer(iy,jy,1)+zfr(nm1,iy,mxm1,jy,0);
        u(0,iy,nm1,0,jy,mzm1) += xfr(iy,nm1,jy,mzm1,0)+yer(iy,jy,2)+zfr(0,iy,0,jy,1);
        u(nm1,iy,nm1,mxm1,jy,mzm1) += xfr(iy,nm1,jy,mzm1,1)+yer(iy,jy,3)+zfr(nm1,iy,mxm1,jy,1);
      } else if ((iy == 0) && (iz == 0) && (jy > 0) && (jz > 0)) {
        u(0,0,0,0,jy,jz) += u(0,nm1,0,0,jy-1,jz)+u(0,0,nm1,0,jy,jz-1)+u(0,nm1,nm1,0,jy-1,jz-1)+xfr(0,0,jy,jz,0)+xfr(nm1,0,jy-1,jz,0)+xfr(0,nm1,jy,jz-1,0)+xfr(nm1,nm1,jy-1,jz-1,0);
        u(0,nm1,0,0,jy-1,jz) = u(0,0,nm1,0,jy,jz-1) = u(0,nm1,nm1,0,jy-1,jz-1) = u(0,0,0,0,jy,jz);
        u(nm1,0,0,mxm1,jy,jz) += u(nm1,nm1,0,mxm1,jy-1,jz)+u(nm1,0,nm1,mxm1,jy,jz-1)+u(nm1,nm1,nm1,mxm1,jy-1,jz-1)+xfr(0,0,jy,jz,1)+xfr(nm1,0,jy-1,jz,1)+xfr(0,nm1,jy,jz-1,1)+xfr(nm1,nm1,jy-1,jz-1,1);
        u(nm1,nm1,0,mxm1,jy-1,jz) = u(nm1,0,nm1,mxm1,jy,jz-1) = u(nm1,nm1,nm1,mxm1,jy-1,jz-1) = u(nm1,0,0,mxm1,jy,jz);
      } else if ((iy > 0) && (iz == 0) && (jz > 0)) {
        u(0,iy,0,0,jy,jz) += u(0,iy,nm1,0,jy,jz-1)+xfr(iy,0,jy,jz,0)+xfr(iy,nm1,jy,jz-1,0);
        u(0,iy,nm1,0,jy,jz-1) = u(0,iy,0,0,jy,jz);
        u(nm1,iy,0,mxm1,jy,jz) += u(nm1,iy,nm1,mxm1,jy,jz-1)+xfr(iy,0,jy,jz,1)+xfr(iy,nm1,jy,jz-1,1);
        u(nm1,iy,nm1,mxm1,jy,jz-1) = u(nm1,iy,0,mxm1,jy,jz);
      } else if ((iy == 0) && (iz > 0) && (jy > 0)) {
        u(0,0,iz,0,jy,jz) += u(0,nm1,iz,0,jy-1,jz)+xfr(0,iz,jy,jz,0)+xfr(nm1,iz,jy-1,jz,0);
        u(0,nm1,iz,0,jy-1,jz) = u(0,0,iz,0,jy,jz);
        u(nm1,0,iz,mxm1,jy,jz) += u(nm1,nm1,iz,mxm1,jy-1,jz)+xfr(0,iz,jy,jz,1)+xfr(nm1,iz,jy-1,jz,1);
        u(nm1,nm1,iz,mxm1,jy-1,jz) = u(nm1,0,iz,mxm1,jy,jz);
      } else if ((iy > 0) && (iz > 0)) {
        u(0,iy,iz,0,jy,jz) += xfr(iy,iz,jy,jz,0);
        u(nm1,iy,iz,mxm1,jy,jz) += xfr(iy,iz,jy,jz,1);
      }
    }
  },stream_[0]);

#else

  MPI_Waitall(2,reqr_,MPI_STATUSES_IGNORE);

  gpuFor({nm1},{nm1},{mx},{my},GPU_LAMBDA(const int ix, const int iy, const int jx, const int jy) {
    if ((ix == 0) && (iy == 0) && (jx > 0) && (jy > 0)) {
      u(0,0,0,jx,jy,0) += u(nm1,0,0,jx-1,jy,0)+u(0,nm1,0,jx,jy-1,0)+u(nm1,nm1,0,jx-1,jy-1,0)+zfr(0,0,jx,jy,0)+zfr(nm1,0,jx-1,jy,0)+zfr(0,nm1,jx,jy-1,0)+zfr(nm1,nm1,jx-1,jy-1,0);
      u(nm1,0,0,jx-1,jy,0) = u(0,nm1,0,jx,jy-1,0) = u(nm1,nm1,0,jx-1,jy-1,0) = u(0,0,0,jx,jy,0);
      u(0,0,nm1,jx,jy,mzm1) += u(nm1,0,nm1,jx-1,jy,mzm1)+u(0,nm1,nm1,jx,jy-1,mzm1)+u(nm1,nm1,nm1,jx-1,jy-1,mzm1)+zfr(0,0,jx,jy,1)+zfr(nm1,0,jx-1,jy,1)+zfr(0,nm1,jx,jy-1,1)+zfr(nm1,nm1,jx-1,jy-1,1);
      u(nm1,0,nm1,jx-1,jy,mzm1) = u(0,nm1,nm1,jx,jy-1,mzm1) = u(nm1,nm1,nm1,jx-1,jy-1,mzm1) = u(0,0,nm1,jx,jy,mzm1);
    } else if ((ix > 0) && (iy == 0) && (jy > 0)) {
      u(ix,0,0,jx,jy,0) += u(ix,nm1,0,jx,jy-1,0)+zfr(ix,0,jx,jy,0)+zfr(ix,nm1,jx,jy-1,0);
      u(ix,nm1,0,jx,jy-1,0) = u(ix,0,0,jx,jy,0);
      u(ix,0,nm1,jx,jy,mzm1) += u(ix,nm1,nm1,jx,jy-1,mzm1)+zfr(ix,0,jx,jy,1)+zfr(ix,nm1,jx,jy-1,1);
      u(ix,nm1,nm1,jx,jy-1,mzm1) = u(ix,0,nm1,jx,jy,mzm1);
    } else if ((ix == 0) && (iy > 0) && (jx > 0)) {
      u(0,iy,0,jx,jy,0) += u(nm1,iy,0,jx-1,jy,0)+zfr(0,iy,jx,jy,0)+zfr(nm1,iy,jx-1,jy,0);
      u(nm1,iy,0,jx-1,jy,0) = u(0,iy,0,jx,jy,0);
      u(0,iy,nm1,jx,jy,mzm1) += u(nm1,iy,nm1,jx-1,jy,mzm1)+zfr(0,iy,jx,jy,1)+zfr(nm1,iy,jx-1,jy,1);
      u(nm1,iy,nm1,jx-1,jy,mzm1) = u(0,iy,nm1,jx,jy,mzm1);
    } else if ((ix > 0) && (iy > 0)) {
      u(ix,iy,0,jx,jy,0) += zfr(ix,iy,jx,jy,0);
      u(ix,iy,nm1,jx,jy,mzm1) += zfr(ix,iy,jx,jy,1);
    }
  },stream_[0]);

  MPI_Waitall(6,reqr_+2,MPI_STATUSES_IGNORE);

  gpuFor({nm1},{nm1},{mx},{mz},GPU_LAMBDA(const int ix, const int iz, const int jx, const int jz) {
    if ((ix == 0) && (iz == 0) && (jx > 0) && (jz == 0)) {
      u(0,0,0,jx,0,0) += u(nm1,0,0,jx-1,0,0)+xer(0,jx,0)+xer(nm1,jx-1,0)+yfr(0,0,jx,0,0)+yfr(nm1,0,jx-1,0,0)+zfr(0,0,jx,0,0)+zfr(nm1,0,jx-1,0,0);
      u(nm1,0,0,jx-1,0,0) = u(0,0,0,jx,0,0);
      u(0,nm1,0,jx,mym1,0) += u(nm1,nm1,0,jx-1,mym1,0)+xer(0,jx,1)+xer(nm1,jx-1,1)+yfr(0,0,jx,0,1)+yfr(nm1,0,jx-1,0,1)+zfr(0,nm1,jx,mym1,0)+zfr(nm1,nm1,jx-1,mym1,0);
      u(nm1,nm1,0,jx-1,mym1,0) = u(0,nm1,0,jx,mym1,0);
      u(0,0,nm1,jx,0,mzm1) += u(nm1,0,nm1,jx-1,0,mzm1)+xer(0,jx,2)+xer(nm1,jx-1,2)+yfr(0,nm1,jx,mzm1,0)+yfr(nm1,nm1,jx-1,mzm1,0)+zfr(0,0,jx,0,1)+zfr(nm1,0,jx-1,0,1);
      u(nm1,0,nm1,jx-1,0,mzm1) = u(0,0,nm1,jx,0,mzm1);
      u(0,nm1,nm1,jx,mym1,mzm1) += u(nm1,nm1,nm1,jx-1,mym1,mzm1)+xer(0,jx,3)+xer(nm1,jx-1,3)+yfr(0,nm1,jx,mzm1,1)+yfr(nm1,nm1,jx-1,mzm1,1)+zfr(0,nm1,jx,mym1,1)+zfr(nm1,nm1,jx-1,mym1,1);
      u(nm1,nm1,nm1,jx-1,mym1,mzm1) = u(0,nm1,nm1,jx,mym1,mzm1);
    } else if ((ix > 0) && (iz == 0) && (jz == 0)) {
      u(ix,0,0,jx,0,0) += xer(ix,jx,0)+yfr(ix,0,jx,0,0)+zfr(ix,0,jx,0,0);
      u(ix,nm1,0,jx,mym1,0) += xer(ix,jx,1)+yfr(ix,0,jx,0,1)+zfr(ix,nm1,jx,mym1,0);
      u(ix,0,nm1,jx,0,mzm1) += xer(ix,jx,2)+yfr(ix,nm1,jx,mzm1,0)+zfr(ix,0,jx,0,1);
      u(ix,nm1,nm1,jx,mym1,mzm1) += xer(ix,jx,3)+yfr(ix,nm1,jx,mzm1,1)+zfr(ix,nm1,jx,mym1,1);
    } else if ((ix == 0) && (iz == 0) && (jx > 0) && (jz > 0)) {
      u(0,0,0,jx,0,jz) += u(nm1,0,0,jx-1,0,jz)+u(0,0,nm1,jx,0,jz-1)+u(nm1,0,nm1,jx-1,0,jz-1)+yfr(0,0,jx,jz,0)+yfr(nm1,0,jx-1,jz,0)+yfr(0,nm1,jx,jz-1,0)+yfr(nm1,nm1,jx-1,jz-1,0);
      u(nm1,0,0,jx-1,0,jz) = u(0,0,nm1,jx,0,jz-1) = u(nm1,0,nm1,jx-1,0,jz-1) = u(0,0,0,jx,0,jz);
      u(0,nm1,0,jx,mym1,jz) += u(nm1,nm1,0,jx-1,mym1,jz)+u(0,nm1,nm1,jx,mym1,jz-1)+u(nm1,nm1,nm1,jx-1,mym1,jz-1)+yfr(0,0,jx,jz,1)+yfr(nm1,0,jx-1,jz,1)+yfr(0,nm1,jx,jz-1,1)+yfr(nm1,nm1,jx-1,jz-1,1);
      u(nm1,nm1,0,jx-1,mym1,jz) = u(0,nm1,nm1,jx,mym1,jz-1) = u(nm1,nm1,nm1,jx-1,mym1,jz-1) = u(0,nm1,0,jx,mym1,jz);
    } else if ((ix == 0) && (iz > 0) && (jx > 0)) {
      u(0,0,iz,jx,0,jz) += u(nm1,0,iz,jx-1,0,jz)+yfr(0,iz,jx,jz,0)+yfr(nm1,iz,jx-1,jz,0);
      u(nm1,0,iz,jx-1,0,jz) = u(0,0,iz,jx,0,jz);
      u(0,nm1,iz,jx,mym1,jz) += u(nm1,nm1,iz,jx-1,mym1,jz)+yfr(0,iz,jx,jz,1)+yfr(nm1,iz,jx-1,jz,1);
      u(nm1,nm1,iz,jx-1,mym1,jz) = u(0,nm1,iz,jx,mym1,jz);
    } else if ((ix > 0) && (iz == 0) && (jz > 0)) {
      u(ix,0,0,jx,0,jz) += u(ix,0,nm1,jx,0,jz-1)+yfr(ix,0,jx,jz,0)+yfr(ix,nm1,jx,jz-1,0);
      u(ix,0,nm1,jx,0,jz-1) = u(ix,0,0,jx,0,jz);
      u(ix,nm1,0,jx,mym1,jz) += u(ix,nm1,nm1,jx,mym1,jz-1)+yfr(ix,0,jx,jz,1)+yfr(ix,nm1,jx,jz-1,1);
      u(ix,nm1,nm1,jx,mym1,jz-1) = u(ix,nm1,0,jx,mym1,jz);
    } else if ((ix > 0) && (iz > 0)) {
      u(ix,0,iz,jx,0,jz) += yfr(ix,iz,jx,jz,0);
      u(ix,nm1,iz,jx,mym1,jz) += yfr(ix,iz,jx,jz,1);
    }
  },stream_[0]);

  MPI_Waitall(18,reqr_+8,MPI_STATUSES_IGNORE);

  gpuFor({nm1},{nm1},{my},{mz},GPU_LAMBDA(const int iy, const int iz, const int jy, const int jz) {
    if ((iy == 0) && (iz == 0) && (jy == 0) && (jz == 0)) {
      u(0,0,0,0,0,0) += cornerr(0,0)+xer(0,0,0)+yer(0,0,0)+zer(0,0,0)+xfr(0,0,0,0,0)+yfr(0,0,0,0,0)+zfr(0,0,0,0,0);
      u(nm1,0,0,mxm1,0,0) += cornerr(0,1)+xer(nm1,mxm1,0)+yer(0,0,1)+zer(0,0,1)+xfr(0,0,0,0,1)+yfr(nm1,0,mxm1,0,0)+zfr(nm1,0,mxm1,0,0);
      u(0,nm1,0,0,mym1,0) += cornerr(0,2)+xer(0,0,1)+yer(nm1,mym1,0)+zer(0,0,2)+xfr(nm1,0,mym1,0,0)+yfr(0,0,0,0,1)+zfr(0,nm1,0,mym1,0);
      u(nm1,nm1,0,mxm1,mym1,0) += cornerr(0,3)+xer(nm1,mxm1,1)+yer(nm1,mym1,1)+zer(0,0,3)+xfr(nm1,0,mym1,0,1)+yfr(nm1,0,mxm1,0,1)+zfr(nm1,nm1,mxm1,mym1,0);
      u(0,0,nm1,0,0,mzm1) += cornerr(0,4)+xer(0,0,2)+yer(0,0,2)+zer(nm1,mzm1,0)+xfr(0,nm1,0,mzm1,0)+yfr(0,nm1,0,mzm1,0)+zfr(0,0,0,0,1);
      u(nm1,0,nm1,mxm1,0,mzm1) += cornerr(0,5)+xer(nm1,mxm1,2)+yer(0,0,3)+zer(nm1,mzm1,1)+xfr(0,nm1,0,mzm1,1)+yfr(nm1,nm1,mxm1,mzm1,0)+zfr(nm1,0,mxm1,0,1);
      u(0,nm1,nm1,0,mym1,mzm1) += cornerr(0,6)+xer(0,0,3)+yer(nm1,mym1,2)+zer(nm1,mzm1,2)+xfr(nm1,nm1,mym1,mzm1,0)+yfr(0,nm1,0,mzm1,1)+zfr(0,nm1,0,mym1,1);
      u(nm1,nm1,nm1,mxm1,mym1,mzm1) += cornerr(0,7)+xer(nm1,mxm1,3)+yer(nm1,mym1,3)+zer(nm1,mzm1,3)+xfr(nm1,nm1,mym1,mzm1,1)+yfr(nm1,nm1,mxm1,mzm1,1)+zfr(nm1,nm1,mxm1,mym1,1);
    } else if ((iy == 0) && (iz == 0) && (jy == 0) && (jz > 0)) {
      u(0,0,0,0,0,jz) += u(0,0,nm1,0,0,jz-1)+xfr(0,0,0,jz,0)+xfr(0,nm1,0,jz-1,0)+yfr(0,0,0,jz,0)+yfr(0,nm1,0,jz-1,0)+zer(0,jz,0)+zer(nm1,jz-1,0);
      u(0,0,nm1,0,0,jz-1) = u(0,0,0,0,0,jz);
      u(nm1,0,0,mxm1,0,jz) += u(nm1,0,nm1,mxm1,0,jz-1)+xfr(0,0,0,jz,1)+xfr(0,nm1,0,jz-1,1)+yfr(nm1,0,mxm1,jz,0)+yfr(nm1,nm1,mxm1,jz-1,0)+zer(0,jz,1)+zer(nm1,jz-1,1);
      u(nm1,0,nm1,mxm1,0,jz-1) = u(nm1,0,0,mxm1,0,jz);
      u(0,nm1,0,0,mym1,jz) += u(0,nm1,nm1,0,mym1,jz-1)+xfr(nm1,0,mym1,jz,0)+xfr(nm1,nm1,mym1,jz-1,0)+yfr(0,0,0,jz,1)+yfr(0,nm1,0,jz-1,1)+zer(0,jz,2)+zer(nm1,jz-1,2);
      u(0,nm1,nm1,0,mym1,jz-1) = u(0,nm1,0,0,mym1,jz);
      u(nm1,nm1,0,mxm1,mym1,jz) += u(nm1,nm1,nm1,mxm1,mym1,jz-1)+xfr(nm1,0,mym1,jz,1)+xfr(nm1,nm1,mym1,jz-1,1)+yfr(nm1,0,mxm1,jz,1)+yfr(nm1,nm1,mxm1,jz-1,1)+zer(0,jz,3)+zer(nm1,jz-1,3);
      u(nm1,nm1,nm1,mxm1,mym1,jz-1) = u(nm1,nm1,0,mxm1,mym1,jz);
    } else if ((iy == 0) && (iz == 0) && (jy > 0) && (jz == 0)) {
      u(0,0,0,0,jy,0) += u(0,nm1,0,0,jy-1,0)+xfr(0,0,jy,0,0)+xfr(nm1,0,jy-1,0,0)+yer(0,jy,0)+yer(nm1,jy-1,0)+zfr(0,0,0,jy,0)+zfr(0,nm1,0,jy-1,0);
      u(0,nm1,0,0,jy-1,0) = u(0,0,0,0,jy,0);
      u(nm1,0,0,mxm1,jy,0) += u(nm1,nm1,0,mxm1,jy-1,0)+xfr(0,0,jy,0,1)+xfr(nm1,0,jy-1,0,1)+yer(0,jy,1)+yer(nm1,jy-1,1)+zfr(nm1,0,mxm1,jy,0)+zfr(nm1,nm1,mxm1,jy-1,0);
      u(nm1,nm1,0,mxm1,jy-1,0) = u(nm1,0,0,mxm1,jy,0);
      u(0,0,nm1,0,jy,mzm1) += u(0,nm1,nm1,0,jy-1,mzm1)+xfr(0,nm1,jy,mzm1,0)+xfr(nm1,nm1,jy-1,mzm1,0)+yer(0,jy,2)+yer(nm1,jy-1,2)+zfr(0,0,0,jy,1)+zfr(0,nm1,0,jy-1,1);
      u(0,nm1,nm1,0,jy-1,mzm1) = u(0,0,nm1,0,jy,mzm1);
      u(nm1,0,nm1,mxm1,jy,mzm1) += u(nm1,nm1,nm1,mxm1,jy-1,mzm1)+xfr(0,nm1,jy,mzm1,1)+xfr(nm1,nm1,jy-1,mzm1,1)+yer(0,jy,3)+yer(nm1,jy-1,3)+zfr(nm1,0,mxm1,jy,1)+zfr(nm1,nm1,mxm1,jy-1,1);
      u(nm1,nm1,nm1,mxm1,jy-1,mzm1) = u(nm1,0,nm1,mxm1,jy,mzm1);
    } else if ((iy == 0) && (iz > 0) && (jy == 0)) {
      u(0,0,iz,0,0,jz) += xfr(0,iz,0,jz,0)+yfr(0,iz,0,jz,0)+zer(iz,jz,0);
      u(nm1,0,iz,mxm1,0,jz) += xfr(0,iz,0,jz,1)+yfr(nm1,iz,mxm1,jz,0)+zer(iz,jz,1);
      u(0,nm1,iz,0,mym1,jz) += xfr(nm1,iz,mym1,jz,0)+yfr(0,iz,0,jz,1)+zer(iz,jz,2);
      u(nm1,nm1,iz,mxm1,mym1,jz) += xfr(nm1,iz,mym1,jz,1)+yfr(nm1,iz,mxm1,jz,1)+zer(iz,jz,3);
    } else if ((iy > 0) && (iz == 0) && (jz == 0)) { 
      u(0,iy,0,0,jy,0) += xfr(iy,0,jy,0,0)+yer(iy,jy,0)+zfr(0,iy,0,jy,0);
      u(nm1,iy,0,mxm1,jy,0) += xfr(iy,0,jy,0,1)+yer(iy,jy,1)+zfr(nm1,iy,mxm1,jy,0);
      u(0,iy,nm1,0,jy,mzm1) += xfr(iy,nm1,jy,mzm1,0)+yer(iy,jy,2)+zfr(0,iy,0,jy,1);
      u(nm1,iy,nm1,mxm1,jy,mzm1) += xfr(iy,nm1,jy,mzm1,1)+yer(iy,jy,3)+zfr(nm1,iy,mxm1,jy,1);
    } else if ((iy == 0) && (iz == 0) && (jy > 0) && (jz > 0)) {
      u(0,0,0,0,jy,jz) += u(0,nm1,0,0,jy-1,jz)+u(0,0,nm1,0,jy,jz-1)+u(0,nm1,nm1,0,jy-1,jz-1)+xfr(0,0,jy,jz,0)+xfr(nm1,0,jy-1,jz,0)+xfr(0,nm1,jy,jz-1,0)+xfr(nm1,nm1,jy-1,jz-1,0);
      u(0,nm1,0,0,jy-1,jz) = u(0,0,nm1,0,jy,jz-1) = u(0,nm1,nm1,0,jy-1,jz-1) = u(0,0,0,0,jy,jz);
      u(nm1,0,0,mxm1,jy,jz) += u(nm1,nm1,0,mxm1,jy-1,jz)+u(nm1,0,nm1,mxm1,jy,jz-1)+u(nm1,nm1,nm1,mxm1,jy-1,jz-1)+xfr(0,0,jy,jz,1)+xfr(nm1,0,jy-1,jz,1)+xfr(0,nm1,jy,jz-1,1)+xfr(nm1,nm1,jy-1,jz-1,1);
      u(nm1,nm1,0,mxm1,jy-1,jz) = u(nm1,0,nm1,mxm1,jy,jz-1) = u(nm1,nm1,nm1,mxm1,jy-1,jz-1) = u(nm1,0,0,mxm1,jy,jz);
    } else if ((iy > 0) && (iz == 0) && (jz > 0)) {
      u(0,iy,0,0,jy,jz) += u(0,iy,nm1,0,jy,jz-1)+xfr(iy,0,jy,jz,0)+xfr(iy,nm1,jy,jz-1,0);
      u(0,iy,nm1,0,jy,jz-1) = u(0,iy,0,0,jy,jz);
      u(nm1,iy,0,mxm1,jy,jz) += u(nm1,iy,nm1,mxm1,jy,jz-1)+xfr(iy,0,jy,jz,1)+xfr(iy,nm1,jy,jz-1,1);
      u(nm1,iy,nm1,mxm1,jy,jz-1) = u(nm1,iy,0,mxm1,jy,jz);
    } else if ((iy == 0) && (iz > 0) && (jy > 0)) {
      u(0,0,iz,0,jy,jz) += u(0,nm1,iz,0,jy-1,jz)+xfr(0,iz,jy,jz,0)+xfr(nm1,iz,jy-1,jz,0);
      u(0,nm1,iz,0,jy-1,jz) = u(0,0,iz,0,jy,jz);
      u(nm1,0,iz,mxm1,jy,jz) += u(nm1,nm1,iz,mxm1,jy-1,jz)+xfr(0,iz,jy,jz,1)+xfr(nm1,iz,jy-1,jz,1);
      u(nm1,nm1,iz,mxm1,jy-1,jz) = u(nm1,0,iz,mxm1,jy,jz);
    } else if ((iy > 0) && (iz > 0)) {
      u(0,iy,iz,0,jy,jz) += xfr(iy,iz,jy,jz,0);
      u(nm1,iy,iz,mxm1,jy,jz) += xfr(iy,iz,jy,jz,1);
    }
  },stream_[0]);

#endif

  // finish sends

  MPI_Waitall(26,reqs_,MPI_STATUSES_IGNORE);
  CHECK(gpuStreamSynchronize(stream_[1]));
  CHECK(gpuStreamSynchronize(stream_[0]));
}
