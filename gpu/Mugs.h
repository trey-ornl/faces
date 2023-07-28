#pragma once

#include "Array.h"

class Mugs {
  public:
    Mugs(int id, int lx, int ly, int lz, int mx, int my, int mz, int n);
    void share(Array<double,6> &u, bool compute = true);

  protected:
    int kx_,ky_,kz_; // coordinates of this task
    int lx_,ly_,lz_; // number of parallel tasks in each dimension
    int mx_,my_,mz_; // local number of elements in each dimension
    int n_; // size of element in each dimension

    int rank_,size_;
    int icorner_[8],iedge_[12],iface_[6],nedge_[3],nface_[3];
    MPI_Request rreq_[26],sreq_[26];

    double rcorner_[8],scorner_[8];
    Array<double,3> rxedge_,ryedge_,rzedge_,sxedge_,syedge_,szedge_;
    Array<double,5> rxface_,ryface_,rzface_,sxface_,syface_,szface_;

    int neighbor(int dx, int dy, int dz) const;
};
