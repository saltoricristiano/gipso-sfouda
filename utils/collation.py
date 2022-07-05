import torch
import MinkowskiEngine as ME


# def minkowski_collate_fn(list_data):
#     r"""
#     Collation function for MinkowskiEngine.SparseTensor that creates batched
#     coordinates given a list of dictionaries.
#     """
#     coordinates_batch, features_batch, labels_batch = ME.utils.batch_sparse_collate(
#         [d["coordinates"] for d in list_data],
#         [d["features"] for d in list_data],
#         [d["labels"] for d in list_data])
#     return {
#         "coordinates": coordinates_batch,
#         "features": features_batch,
#         "labels": labels_batch,
#     }

class CollateFN:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        # batched_coords = [d['coordinates'] for d in list_data]
        # batched_coords = ME.utils.batched_coordinates(batched_coords, dtype=torch.float32)

        list_data = [(d["coordinates"].to(self.device), d["features"].to(self.device), d["labels"]) for d in list_data]

        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_data)
        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch}


class CollateMixed:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        actual_matches_list = [d["matches"] for d in list_data]
        next_matches_list = [d["next_matches"] for d in list_data]

        matches_list = [None] * (len(actual_matches_list)+len(next_matches_list))
        matches_list[::2] = actual_matches_list
        matches_list[1::2] = next_matches_list

        matches = torch.cat(matches_list)

        match_idx0 = torch.where(matches == 0)[0]
        match_idx1 = torch.where(matches == 1)[0]

        next_list_data = [(d["next_coordinates"].to(self.device), d["next_features"].to(self.device), d["next_labels"].to(self.device)) for d in list_data]
        actual_list_data = [(d["coordinates"].to(self.device), d["features"].to(self.device), d["labels"].to(self.device)) for d in list_data]

        batch_list = [None] * (len(next_list_data)+len(actual_list_data))
        batch_list[::2] = actual_list_data
        batch_list[1::2] = actual_list_data

        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32)(batch_list)

        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch,
                "matches": matches,
                "fwd_match": match_idx0,
                "bck_match": match_idx1}


class CollateSeparated:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """

        matches_list0 = []
        matches_list1 = []
        num_data = len(list_data)

        list_data0 = []
        list_data1 = []

        list_num_pts0 = []
        list_num_pts1 = []

        list_all = []
        list_selected = []

        list_geometric = []

        start_pts0 = 0
        start_pts1 = 0

        for d in range(num_data):

            # shift for minkowski forward and append
            matches_list0.append(list_data[d]["matches0"] + start_pts0)
            matches_list1.append(list_data[d]["matches1"] + start_pts1)

            start_pts0 += list_data[d]["num_pts0"]
            start_pts1 += list_data[d]["num_pts1"]

            list_num_pts0.append(list_data[d]["num_pts0"])
            list_num_pts1.append(list_data[d]["num_pts1"])

            list_data0.append((list_data[d]["coordinates"].to(self.device), list_data[d]["features"].to(self.device), list_data[d]["labels"].to(self.device)))
            list_data1.append((list_data[d]["next_coordinates"].to(self.device), list_data[d]["next_features"].to(self.device), list_data[d]["next_labels"].to(self.device)))

            list_all.append(list_data[d]["coordinates_all"].to(self.device))

            list_geometric.append(list_data[d]["geometric_feats"].to(self.device))

            list_selected.append(list_data[d]["sampled_idx"].to(self.device))

        # concatenate
        matches_list0 = torch.cat(matches_list0)
        matches_list1 = torch.cat(matches_list1)

        # sparse collation for t0
        coordinates_batch0, features_batch0, labels_batch0 = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                       device=self.device)(list_data0)
        # sparse collation for t1
        coordinates_batch1, features_batch1, labels_batch1 = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                       device=self.device)(list_data1)

        return {"coordinates0": coordinates_batch0,
                "features0": features_batch0,
                "labels0": labels_batch0,
                "geometric_features0": list_geometric,
                "coordinates1": coordinates_batch1,
                "features1": features_batch1,
                "labels1": labels_batch1,
                "matches0": matches_list0,
                "matches1": matches_list1,
                "num_pts0": list_num_pts0,
                "num_pts1": list_num_pts1,
                "coordinates_all": list_all,
                "sampled_idx": list_selected}


class CollateStream:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        # batched_coords = [d['coordinates'] for d in list_data]
        # batched_coords = ME.utils.batched_coordinates(batched_coords, dtype=torch.float32)

        batch_data = []
        batch_global = []

        for d in list_data:
            batch_data.append((d["coordinates"], d["features"], d["labels"]))
            batch_global.append(d['global_points'])

        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(batch_data)
        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch,
                "global_points": batch_global}
