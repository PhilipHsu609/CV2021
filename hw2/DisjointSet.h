#ifndef DISJOINTSET_H
#define DISJOINTSET_H

#include <numeric>
#include <vector>

class DisjointSet {
   public:
    DisjointSet(int n) {
        parent.resize(n + 1);
        sz.resize(n + 1, 1);
        std::iota(begin(parent), end(parent), 0);
    }

    int find_parent(int x) {
        if (x != parent[x])
            parent[x] = find_parent(parent[x]);
        return parent[x];
    }

    void union_set(int x, int y) {
        int a = find_parent(x);
        int b = find_parent(y);

        if (sz[a] < sz[b]) std::swap(a, b);
        parent[b] = a;
        sz[a] += sz[b];
    }

   private:
    std::vector<int> parent;
    std::vector<int> sz;
};

#endif