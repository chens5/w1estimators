#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>

namespace py = pybind11;

namespace pde {

const double EPS = 1e-8;
const double EPS2 = 1e-5;
const double EPS3 = 1e-3;

inline int32_t sign(float x) {
  if (fabs(x) < EPS) {
    throw std::logic_error("computing sign of ~0");
  }
  if (x > 0) return 1;
  return -1;
}

class PDEstimators {
 public:
  using NumPyFloatArray = py::array_t<float, py::array::c_style>;
  using NumPyIntArray = py::array_t<int32_t, py::array::c_style>;

  using EigenVector = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
  using EigenMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Matrix = Eigen::Map<EigenMatrix>;

  PDEstimators() : stage(0) {}


  void load_vocabulary(NumPyFloatArray points) {
    if (stage != 0) {
      throw std::logic_error(
          "load_vocabulary() should be called once in the beginning");
    }
    stage = 1;
    py::buffer_info buf = points.request();
    if (buf.ndim != 2) {
      throw std::logic_error(
          "load_vocabulary() expects a two-dimensional NumPy array");
    }
    // number of points
    auto n = buf.shape[0];
    // dimension of the points - in our case - it is always 2
    auto d = buf.shape[1];
    // make a dictionary. each row contains one word
    dictionary = std::make_unique<Matrix>(static_cast<float *>(buf.ptr), n, d);
    // minimum point
    auto cmin = std::numeric_limits<float>::max();
    // maximum point
    auto cmax = std::numeric_limits<float>::min();
    for (ssize_t i = 0; i < n; ++i) {
      for (ssize_t j = 0; j < d; ++j) {
        cmin = std::min(cmin, (*dictionary)(i, j));
        cmax = std::max(cmax, (*dictionary)(i, j));
      }
    }
    // how large should the bounding box be
    auto delta = cmax - cmin;
    cmin -= delta;
    delta0 = 2.0*delta;
    // generate random shift
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<float> shift_gen(0.0, delta);
    std::vector<std::pair<float, float>> bounding_box;
    // shifting in all dimensions
    for (ssize_t i = 0; i < d; ++i) {
      auto s = shift_gen(gen);
      bounding_box.push_back(std::make_pair(cmin + s, cmax + s));
    }
    // At the end of this, we get a randomly shifted box
    std::vector<int32_t> all;
    for (ssize_t i = 0; i < n; ++i) {
      all.push_back(i);
    }
    // There are n leaves so make sure to resize the leaf list
    leaf_pd.resize(n);
    // now build the quadtree
    std::cout << "size of bounding box: " << delta << '\n';
    build_quadtree_pd(all, bounding_box, 0, -1);
    num_queries = 0;
    marked.resize(parents.size());
    for (auto &x : marked) {
      x = -1;
    }
    node_id.resize(parents.size());
  }

  void load_diagrams(const std::vector<std::vector<std::pair<int32_t, float>>> &dataset) {
    for (auto &measure : dataset) {
      dataset_embedding.push_back(compute_embedding(measure));
    }
  }
public:
  std::vector<std::pair<int32_t, bool>> parents;
  std::vector<int32_t> marked;
  int32_t num_queries;
  std::vector<int32_t> node_id;
  std::vector<int32_t> id_node;
  std::vector<std::vector<std::pair<int32_t, bool>>> subtree;
  std::vector<float> delta_node;
  std::unique_ptr<Matrix> dictionary;
  std::vector<std::vector<std::pair<int32_t, float>>> dataset_embedding;
  std::vector<std::vector<std::pair<int32_t, float>>> raw_dataset;
  int32_t stage;
  float delta0;

  // New leaf nodes for quadtree
  std::vector<int32_t> leaf_pd;
  std::vector<std::pair<int32_t, bool>> unleaf_pd;
  std::vector<std::vector<std::pair<float, int32_t>>> excess_pd;
  std::vector<std::pair<float, int32_t>> demand_reg;
  std::vector<std::pair<float, int32_t>> supply_reg;

  std::vector<bool> s_tree_d;

  void print_tree_details() {
    int32_t parent_size = parents.size();
    int32_t leaf_size = leaf_pd.size();
    std::cout << "SIZE OF BOUNDING BOX: "<< delta0 << '\n';
    std::cout << "NODES IN TREE: \n";
    for(int32_t i = 0; i < parent_size; ++i) {
      std::cout << parents[i].first << ' ';
      if (parents[i].second) {
        std::cout << "diagonal\n";
      }
      else {
        std::cout << '\n';
      }
    }
    std::cout << "LEAVES: \n";
    for (int32_t i = 0; i < leaf_size; ++i) {
      std::cout << "point: " << i << " in node: "<< leaf_pd[i] << '\n';
    }
  }

  float quadtree_query_pair(const int32_t a,
  const int32_t b){
    auto query_embedding = dataset_embedding[a];
    auto point_embedding = dataset_embedding[b];
    std::vector<std::pair<int32_t, float>> q_save;
    std::vector<std::pair<int32_t, float>> p_save;

    float score = 0.0;
    size_t qp = 0;
    size_t dp = 0;
    int32_t parent_size = parents.size();
    for(std::vector<std::pair<int32_t, float>>::reverse_iterator it = query_embedding.rbegin();
  it != query_embedding.rend(); ++it) {
    if ((*it).first >= parent_size) {
      score += query_embedding.back().second;
      query_embedding.pop_back();
      q_save.push_back((*it));
    }
    if((*it).first < parent_size) {
      break;
    }
  }
  for(std::vector<std::pair<int32_t, float>>::reverse_iterator it = point_embedding.rbegin();
  it != point_embedding.rend(); ++it) {
    if ((*it).first >= parent_size) {
      score += point_embedding.back().second;
      point_embedding.pop_back();
      p_save.push_back((*it));
    }
    if((*it).first < parent_size) {
      break;
    }
  }
    while( qp < query_embedding.size() || dp < point_embedding.size()){
      if (qp == query_embedding.size()) {
        if(!parents[ point_embedding[dp].first].second) {
          score += point_embedding[dp].second;
        }
        ++dp;
      } else if (dp == point_embedding.size()) {
        if ( !parents[query_embedding[qp].first].second) {
          score += query_embedding[qp].second;
        }
        ++qp;
      } else if (query_embedding[qp].first < point_embedding[dp].first) {
        if ( !parents[query_embedding[qp].first].second) {
          score += query_embedding[qp].second;
        }
        ++qp;
      } else if (point_embedding[dp].first < query_embedding[qp].first ) {
        if (!parents[point_embedding[dp].first].second){
          score += point_embedding[dp].second;
        }
        ++dp;
      } else if (!parents[point_embedding[dp].first].second && !parents[query_embedding[qp].first].second){
        score +=
            fabs(query_embedding[qp].second - point_embedding[dp].second);
        ++qp;
        ++dp;
      } else {
        ++qp;
        ++dp;
      }

    }
    for(std::vector<std::pair<int32_t, float>>::reverse_iterator it = q_save.rbegin();
      it != q_save.rend(); ++it) {
        query_embedding.push_back((*it));
      }
      for(std::vector<std::pair<int32_t, float>>::reverse_iterator it = p_save.rbegin();
    it != p_save.rend(); ++it) {
      point_embedding.push_back((*it));
    }
        return delta0 * score;

  }

  std::vector<std::pair<int32_t, float>> compute_embedding(
      const std::vector<std::pair<int32_t, float>> &a) {
    std::vector<std::pair<int32_t, float>> result;
    int32_t index = parents.size();
    for (auto x : a) {
      auto id = leaf_pd[x.first];
      int32_t level = 0;
      while (id != -1) {
        ++level;
        id = parents[id].first;
      }
      id = leaf_pd[x.first];
      if (parents[id].second) {
        result.push_back(std::make_pair(index, x.second/(1 << level)));
        index++;
      }
      while (id != -1) {
        --level;
        result.push_back(std::make_pair(id, x.second / (1 << level)));
        id = parents[id].first;
      }
    }
    std::sort(result.begin(), result.end());
    std::vector<std::pair<int32_t, float>> ans;
    for (auto x : result) {
      if (ans.empty() || ans.back().first != x.first) {
        ans.push_back(x);
      } else {
        ans.back().second += x.second;
      }
    }
    return ans;
  }

  void build_quadtree_pd(const std::vector<int32_t> &subset,
                        const std::vector<std::pair<float, float>> &bounding_box,
                        int32_t depth, int32_t parent){

      int32_t node_id(parents.size());

      bool is_diagonal = false;
      float x_coord = bounding_box[0].second;
      float y_coord = bounding_box[1].first;
      if (x_coord > y_coord) {
        is_diagonal = true;
      }
      std::pair<int32_t, bool> p;
      p.second = is_diagonal;
      p.first = parent;

      parents.push_back(p);


      if(subset.size() == 1) {
        leaf_pd[subset[0]] = node_id;
        return;
      }

      int32_t d = dictionary->cols();
      // instantiate mid with size d
      std::vector<float> mid(d);
      for (int32_t i = 0; i < d; ++i) {
        mid[i] = (bounding_box[i].first + bounding_box[i].second) / 2.0;
      }
      std::map<std::vector<uint8_t>, std::vector<int32_t>> parts;
      for (auto ind : subset) {
        std::vector<uint8_t> code((d + 7) / 8, 0);
        for (int32_t i = 0; i < d; ++i) {
          if ((*dictionary)(ind, i) > mid[i]) {
            code[i / 8] |= 1 << (i % 8);
          }
        }
        parts[code].push_back(ind);
      }
      std::vector<std::pair<float, float>> new_bounding_box(d);
      for (const auto &part : parts) {
        for (int32_t i = 0; i < d; ++i) {
          uint8_t bit = (part.first[i / 8] >> (i % 8)) & 1;
          if (bit) {
            new_bounding_box[i] = std::make_pair(mid[i], bounding_box[i].second);
          } else {
            new_bounding_box[i] = std::make_pair(bounding_box[i].first, mid[i]);
          }
        }
        build_quadtree_pd(part.second, new_bounding_box, depth + 1, node_id);
      }

  }

  float flowtree_query(const std::vector<std::pair<int32_t, float>> &a,
                          const std::vector<std::pair<int32_t, float>> &b,
                          const int32_t internal_norm){

    int32_t num_nodes = 0;
    id_node.clear();
    // finding where the nodes where a and b are in the tree
    for (auto x : a) {
      // get the spot in the parents that contains the leaf's parent
      auto id = leaf_pd[x.first];
      while (id != -1) {
        // check if that node is marked
        if (marked[id] != num_queries) {
          // if marked, add it to the list id_node
          // id_node (at the end of this process) will have all the relevant parent nodes
          id_node.push_back(id);
          // node_id[id] counts the number of nodes seen so far at the first time we
          // se a specific parent coded to id
          node_id[id] = num_nodes++;

        }
        marked[id] = num_queries;
        // go up the tree
        id = parents[id].first;
      }
    }
    for (auto x : b) {
      auto id = leaf_pd[x.first];
      while (id != -1) {
        if (marked[id] != num_queries) {
          id_node.push_back(id);
          node_id[id] = num_nodes++;
        }
        marked[id] = num_queries;
        id = parents[id].first;
      }
    }
    // resize subtree so it will contain the correct number of nodes
    if (static_cast<int32_t>(subtree.size()) < num_nodes) {
      subtree.resize(num_nodes);
    }
    if (static_cast<int32_t>(s_tree_d.size()) < num_nodes) {
      s_tree_d.resize(num_nodes);
    }
    // clear each spot in the subtree
    for (int32_t i = 0; i < num_nodes; ++i) {
      subtree[i].clear();
    }

    for (int32_t i = 0; i < num_nodes; ++i) {
      // id_node[i] will contain spots for parents
      // parents[id_node[i]] will contain representation of node
      std::pair<int32_t, bool> u = parents[id_node[i]];
      if (u.first != -1) {
        // node_id[u] = num_nodes at time of add
        // subtree[node_id[u]] the subtree rooted at parent node u
        // this subtree is represented by i
        std::pair<int32_t, bool> to_push;
        to_push.first = i;
        if(u.second) {
          to_push.second = true;
        }
        else {
          to_push.second = false;
        }
        // new (experiment)
        std::pair<int32_t, bool> p = parents[u.first];
        if(p.second) {
          s_tree_d[node_id[u.first]] = true;
        } else {
          s_tree_d[node_id[u.first]] = false;
        }
        // end new
        subtree[node_id[u.first]].push_back(to_push);
      }
    }

    if (static_cast<int32_t>(excess_pd.size()) < num_nodes) {
      excess_pd.resize(num_nodes);
    }

    delta_node.assign(num_nodes, 0.0);
    unleaf_pd.resize(num_nodes);
    // mark each node diagonally?
    for(auto x : a) {
      // what does delta_node do:
      // leaf[x.first] = gets leaf associated with the point x
      // Recall that we identify a leaf by the number of parents it has
      // node_id[leaf[x.first]] = gets num_nodes??
      // delta_node wants to get all the weight associated with the point x
      // pushing all demand up to some delta node
      delta_node[node_id[leaf_pd[x.first]]] += x.second;
      std::pair<int32_t, int32_t> to_unleaf;
      to_unleaf.first = x.first;
      to_unleaf.second = parents[leaf_pd[x.first]].second;
      unleaf_pd[node_id[leaf_pd[x.first]]] = to_unleaf;
    }
    for (auto x : b) {
      delta_node[node_id[leaf_pd[x.first]]] -= x.second;
      std::pair<int32_t, int32_t> to_unleaf;
      to_unleaf.first = x.first;
      to_unleaf.second = parents[leaf_pd[x.first]].second;
      unleaf_pd[node_id[leaf_pd[x.first]]] = to_unleaf;
    }

    float res = run_query_pd(0, node_id[0], internal_norm);
    if (!excess_pd[node_id[0]].empty()) {
      float unassigned = 0.0;
      for (auto x : excess_pd[node_id[0]]) {
        auto point = dictionary->row(x.second);
        EigenVector proj;
        proj.resize(2);
        proj[0] = (point[0] + point[1]) / 2;
        proj[1] = proj[0];
        float dist = 0;
        if (internal_norm == 2) {
          dist = (point - proj).norm();
        } else if (internal_norm == 1) {
          dist = fabs(point[0] - proj[0]) + fabs(point[1] - proj[1]);
        } else {
          dist = std::max(fabs(point[0] - proj[0]), fabs(point[1] - proj[1]));
        }
        res += dist * fabs(x.first);
      }
      if (unassigned > EPS2) {
        throw std::logic_error("too much unassigned flow");
      }
    }
    ++num_queries;
    return res;
  }

  float run_query_pd(int32_t depth, int32_t nd, int32_t internal_norm) {
    float res = 0.0;
    for (auto x : subtree[nd]) {
      res += run_query_pd(depth + 1, x.first, internal_norm);
    }
    excess_pd[nd].clear();
    if (subtree[nd].empty()) {
      if (fabs(delta_node[nd]) > EPS) {
        if (unleaf_pd[nd].second){
          auto point = dictionary->row(unleaf_pd[nd].first);
          EigenVector proj;
          proj.resize(2);
          proj[0] = (point[0] + point[1]) / 2;
          proj[1] = proj[0];
          float dist = 0;
          if (internal_norm == 2) {
            dist = (point - proj).norm();
          } else if (internal_norm == 1) {
            dist = fabs(point[0] - proj[0]) + fabs(point[1] - proj[1]);
          } else {
            dist = std::max(fabs(point[0] - proj[0]), fabs(point[1] - proj[1]));
          }
          res += dist * fabs(delta_node[nd]);
        }
        else{
          excess_pd[nd].push_back(std::make_pair(delta_node[nd], unleaf_pd[nd].first));
        }

      }
    } else {
      bool diagonal_node = false;
      if (s_tree_d[nd]) {
        diagonal_node = true;
      }
      for (auto x : subtree[nd]) {
        if(x.second) {
          diagonal_node = true;
        }
        if (excess_pd[x.first].empty()) {
          continue;
        }

        for (auto y : excess_pd[x.first]) {
          if (sign(y.first)  == -1) {
            demand_reg.push_back(y);
          }
          else {
            supply_reg.push_back(y);
          }
        }

      }
      // There is still demand and supply that cannot be matched to diagonal
      while(!demand_reg.empty() && !supply_reg.empty()) {
        auto u = demand_reg.back();
        auto v = supply_reg.back();
        auto p1 = dictionary->row(u.second);
        auto p2 = dictionary->row(v.second);
        float dist = 0;
        if (internal_norm == 2) {
          dist = (p1 - p2).norm();
        } else if (internal_norm == 1) {
          dist = fabs(p1[0] - p2[0]) + fabs(p1[1] - p2[1]);
        } else {
          dist = std::max(fabs(p1[0] - p2[0]), fabs(p1[1] - p2[1]));
        }

        // equal suppy and demand
        if (fabs(u.first + v.first) < EPS) {
          demand_reg.pop_back();
          supply_reg.pop_back();
          res += dist * fabs(u.first);
        } else if (fabs(u.first) < fabs(v.first)) {
          demand_reg.pop_back();
          supply_reg.back().first += u.first;
          res += dist * fabs(u.first);
        } else {
          supply_reg.pop_back();
          demand_reg.back().first += v.first;
          res += dist * fabs(v.first);
        }
      }

      while(!demand_reg.empty() && diagonal_node) {
        auto u = demand_reg.back();
        demand_reg.pop_back();

        auto point = dictionary->row(u.second);
        EigenVector proj;
        proj.resize(2);
        proj[0] = (point[0] + point[1]) / 2;
        proj[1] = proj[0];
        float dist = 0;
        if (internal_norm == 2) {
          dist = (point - proj).norm();
        } else if (internal_norm == 1) {
          dist = fabs(point[0] - proj[0]) + fabs(point[1] - proj[1]);
        } else {
          dist = std::max(fabs(point[0] - proj[0]), fabs(point[1] - proj[1]));
        }
        res += dist *fabs(u.first);
      }
      while(!supply_reg.empty() && diagonal_node) {
        auto u = supply_reg.back();
        supply_reg.pop_back();
        auto point = dictionary->row(u.second);
        EigenVector proj;
        proj.resize(2);
        proj[0] = (point[0] + point[1]) / 2;
        proj[1] = proj[0];
        float dist = 0;
        if (internal_norm == 2) {
          dist = (point - proj).norm();
        } else if (internal_norm == 1) {
          dist = fabs(point[0] - proj[0]) + fabs(point[1] - proj[1]);
        } else {
          dist = std::max(fabs(point[0] - proj[0]), fabs(point[1] - proj[1]));
        }
        res += dist * fabs(u.first);
      }
      if (!supply_reg.empty()) {
        supply_reg.swap(excess_pd[nd]);
      }
      if (!demand_reg.empty()) {
        demand_reg.swap(excess_pd[nd]);
      }
      supply_reg.clear();
      demand_reg.clear();
    }
    return res;
  }



  void check_stage() {
    if (stage != 2) {
      throw std::logic_error(
          "need to call load_vocabulary() and load_dataset() first");
    }
  }

  template <typename T>
  void check_dimension(T x) {
    auto buf = x.request();
    if (buf.ndim != 1) {
      throw std::logic_error(
          "input_ids, output_ids, output_scores must be one-dimensional");
    }
  }

  template <typename T>
  ssize_t get_length(T x) {
    return x.request().shape[0];
  }



};
}  // namespace ote

PYBIND11_MODULE(pd_estimators, m) {
  using pde::PDEstimators;
  py::class_<PDEstimators>(m, "PDEstimators")
      .def(py::init<>())
      .def("load_vocabulary", &PDEstimators::load_vocabulary)
      .def("load_diagrams", &PDEstimators::load_diagrams)
      .def("print_tree_details", &PDEstimators::print_tree_details)
      .def("quadtree_query_pair", &PDEstimators::quadtree_query_pair)
      .def("flowtree_query", &PDEstimators::flowtree_query);
}
