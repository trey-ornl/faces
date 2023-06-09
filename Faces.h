#pragma once

#include "DArray.h"


class Faces
{
  public:
    using Double2D = DArray<double,2>;
    using Double3D = DArray<double,3>;
    using Double5D = DArray<double,5>;
    using Double6D = DArray<double,6>;

    Faces(int rank, int lx, int ly, int lz, int mx, int my, int mz, int n);
    ~Faces();
    void share(Double6D &u, bool compute = true);

  protected:
    int kx_,ky_,kz_;
    int lx_,ly_,lz_;
    int mx_,my_,mz_;
    int n_;

    int rank_,size_;
    int icorner_[8],iedge_[12],iface_[6],nedge_[3],nface_[3];
    MPI_Request reqr_[26],reqs_[26];

    static constexpr int nStreams_ = 2;
    hipStream_t stream_[nStreams_];

    Double2D cornerr_, corners_;
    Double3D xer_,yer_,zer_;
    Double3D xes_,yes_,zes_;

    Double5D xfr_,yfr_,zfr_;
    Double5D xfs_,yfs_,zfs_;

    int neighbor(int dx, int dy, int dz) const;
};
