#pragma once

#include "DArray.h"


class Faces
{
  public:
    using Double5D = DArray<double,5>;
    using Double6D = DArray<double,6>;

    Faces(int rank, int lx, int ly, int lz, int mx, int my, int mz, int n);
    void share(Double6D &u, bool compute = true);

  protected:
    int kx_,ky_,kz_;
    int lx_,ly_,lz_;
    int mx_,my_,mz_;
    int n_;

    int rank_,size_;
    int iface_[6],nface_[3];
    MPI_Request reqr_[6],reqs_[6];

    Double5D xfr_,yfr_,zfr_;
    Double5D xfs_,yfs_,zfs_;

    int neighbor(int dx, int dy, int dz) const;
};
