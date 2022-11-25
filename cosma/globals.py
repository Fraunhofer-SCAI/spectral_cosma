import torch
import numpy as np


# the pooling mask for the padded patches at second level (starts at 4 and not at 0)
PAD_2NDLEVEL_POOLING_MASK = [[],
                             [],
                             [],
                             [4,
                              11, 13,
                              22, 24, 26],
                             [4, # +7
                              11, 13, # +9
                              22, 24, 26, # +11
                              37, 39, 41, 43, # +13
                              56, 58, 60, 62, 64
                              ]]

    
# a helper class to compute the adjacency matrix, pooling mask and edge index
# for patches of different refinement levels
class RegularAdjacencyMatrixCreator:
    def __init__(self):
        self.edge_list = [[0, 1], [0, 2], [1, 2]]
        self.pooling_mask = [0]
        self.num_rows = 2
        self.num_nodes = self.get_num_nodes()
        self.num_refinements = 0
        
    def get_num_nodes(self):
        return int(self.num_rows * (self.num_rows + 1) / 2)
    
    def add_edges(self):
        for low_id in range(self.num_nodes - self.num_rows, self.num_nodes):
            # add vertical edges 
            self.edge_list += [[low_id, low_id + self.num_rows]] + [[low_id, low_id + self.num_rows + 1]]
            # add horizontal edges
            self.edge_list += [[low_id + self.num_rows, low_id + self.num_rows + 1]]
    
    def add_pooling_nodes(self):
        for low_id in range(self.num_nodes - self.num_rows, self.num_nodes):
            if (low_id - (self.num_nodes - self.num_rows)) % 2 == 0:
                self.pooling_mask += [low_id]
    
    def add_row(self):
        self.add_edges()
        self.num_rows += 1
        self.num_nodes = self.get_num_nodes()
        if self.num_rows % 2 == 1:
            self.add_pooling_nodes()
    
    def add_refinement(self):
        self.num_refinements += 1
        initial_num_rows = self.num_rows
        #print('self.num_rows before refinement', self.num_rows, end='')
        while self.num_rows < 2 * initial_num_rows - 1:
            self.add_row()
        #print('; after refinement', self.num_rows)
            
    def get_adj_matrix(self, num_refinements = 3, max_refinements = 3, padding = False):
        for _ in range(num_refinements):
            self.add_refinement()
            
        if padding:
            if num_refinements == max_refinements:
                additional_rows = 6
            elif num_refinements + 1 == max_refinements:
                additional_rows = 3
            else:
                additional_rows = 0
            #print('additional_rows', additional_rows)
            for _ in range(additional_rows):
                self.add_row()
            
        adj_mat = torch.zeros((self.num_nodes, self.num_nodes), dtype = torch.long)
        edge_list = torch.tensor(self.edge_list)
        edge_index = torch.zeros((2, 2 * edge_list.shape[0]), dtype = torch.long)
        
        edge_index[0, :edge_list.shape[0]] = edge_list[:, 0]
        edge_index[0, edge_list.shape[0]:] = edge_list[:, 1]
        edge_index[1, :edge_list.shape[0]] = edge_list[:, 1]
        edge_index[1, edge_list.shape[0]:] = edge_list[:, 0]
        
        adj_mat[edge_index[0], edge_index[1]] = 1
        if (padding and num_refinements + 1 == max_refinements):
            pooling_mask = torch.tensor(PAD_2NDLEVEL_POOLING_MASK[max_refinements], dtype = torch.long)
        else:
            pooling_mask = torch.tensor(self.pooling_mask, dtype = torch.long)
        return adj_mat, edge_index, pooling_mask 
    
    
    
# get the adjacency matrix, edge index and pooling mask
# for every input refinement level up to MAX_REFINEMENT
MAX_INITIAL_REFINEMENT = 4
MIN_INITIAL_REFINEMENT = 3

PAD_ADJ_MATS_RF, PAD_EDGE_INDICES_RF, PAD_POOLING_MASKS_RF = dict(), dict(), dict()
NONPAD_ADJ_MATS_RF, NONPAD_EDGE_INDICES_RF, NONPAD_POOLING_MASKS_RF = dict(), dict(), dict()

for MAX_REFINEMENT in range(MIN_INITIAL_REFINEMENT, MAX_INITIAL_REFINEMENT+1):
        
    PAD_ADJ_MATS_RF[MAX_REFINEMENT], PAD_EDGE_INDICES_RF[MAX_REFINEMENT], PAD_POOLING_MASKS_RF[MAX_REFINEMENT] = [], [], []
    NONPAD_ADJ_MATS_RF[MAX_REFINEMENT], NONPAD_EDGE_INDICES_RF[MAX_REFINEMENT], NONPAD_POOLING_MASKS_RF[MAX_REFINEMENT] = [], [], []

    for refinement in range(MAX_REFINEMENT + 1):
        
        adj_mat, edge_index, pooling_mask = RegularAdjacencyMatrixCreator().get_adj_matrix(num_refinements = refinement, max_refinements = MAX_REFINEMENT, padding = True)
        PAD_ADJ_MATS_RF[MAX_REFINEMENT].append(adj_mat)
        PAD_EDGE_INDICES_RF[MAX_REFINEMENT].append(edge_index)
        PAD_POOLING_MASKS_RF[MAX_REFINEMENT].append(pooling_mask)

        adj_mat, edge_index, pooling_mask = RegularAdjacencyMatrixCreator().get_adj_matrix(num_refinements = refinement, padding = False)
        NONPAD_ADJ_MATS_RF[MAX_REFINEMENT].append(adj_mat)
        NONPAD_EDGE_INDICES_RF[MAX_REFINEMENT].append(edge_index)
        NONPAD_POOLING_MASKS_RF[MAX_REFINEMENT].append(pooling_mask)


        

    
# a mapping to reorder the indices of the original padded patches
PAD_INDEX_MAPPING_KS2 = [[],[],[],
                     # refinement = 3
                     [0, 
                     3, 1, 
                     3, 0, 1,
                     9, 4, 5, 2, 
                     17, 10, 11, 6, 7,
                     27, 18, 19, 12, 13, 8,
                     39, 28, 29, 20, 21, 14, 15,
                     52, 40, 41, 30, 31, 22, 23, 16,
                     65, 53, 54, 42, 43, 32, 33, 24, 25,
                     78, 66, 67, 55, 56, 44, 45, 34, 35, 26,
                     89, 79, 80, 68, 69, 57, 58, 46, 47, 36, 37,
                     98, 90, 91, 81, 82, 70, 71, 59, 60, 48, 49, 38,
                     105, 99, 100, 92, 93, 83, 84, 72, 73, 61, 62, 50, 51,
                     105, 106, 107, 101, 102, 94, 95, 85, 86, 74, 75, 63, 64, 51,
                     106, 110, 110, 108, 109, 103, 104, 96, 97, 87, 88, 76, 77, 77, 64
                    ],
                    # patch_size = 2+np.sum(2**np.arange(args.refine)) +2 *args.kernel_size

                         
                    # refinement = 4     
                    [  0, 
                       3,   1, 
                       3,   0,   1,
                       9,   4,   5,   2, 
                      17,  10,  11,   6,   7,
                      27,  18,  19,  12,  13,   8,
                      39,  28,  29,  20,  21,  14,  15,
                      53,  40,  41,  30,  31,  22,  23,  16,
                      69,  54,  55,  42,  43,  32,  33,  24,  25,
                      87,  70,  71,  56,  57,  44,  45,  34,  35,  26,
                     107,  88,  89,  72,  73,  58,  59,  46,  47,  36,  37,
                     128, 108, 109,  90,  91,  74,  75,  60,  61,  48,  49,  38,
                     149, 129, 130, 110, 111,  92,  93,  76,  77,  62,  63,  50,  51,
                     170, 150, 151, 131, 132, 112, 113,  94,  95,  78,  79,  64,  65,  52,
                     189, 171, 172, 152, 153, 133, 134, 114, 115,  96,  97,  80,  81,  66,  67,
                     206, 190, 191, 173, 174, 154, 155, 135, 136, 116, 117,  98,  99,  82,  83,  68,
                     221, 207, 208, 192, 193, 175, 176, 156, 157, 137, 138, 118, 119, 100, 101,  84,  85, 
                     234, 222, 223, 209, 210, 194, 195, 177, 178, 158, 159, 139, 140, 120, 121, 102, 103,  86,
                     245, 235, 236, 224, 225, 211, 212, 196, 197, 179, 180, 160, 161, 141, 142, 122, 123, 104, 105,
                     254, 246, 247, 237, 238, 226, 227, 213, 214, 198, 199, 181, 182, 162, 163, 143, 144, 124, 125, 106,
                     261, 255, 256, 248, 249, 239, 240, 228, 229, 215, 216, 200, 201, 183, 184, 164, 165, 145, 146, 126, 127,
                     261, 262, 263, 257, 258, 250, 251, 241, 242, 230, 231, 217, 218, 202, 203, 185, 186, 166, 167, 147, 148, 127,
                     262, 266, 266, 264, 265, 259, 260, 252, 253, 243, 244, 232, 233, 219, 220, 204, 205, 187, 188, 168, 169, 169, 148,
                    ]
                    # patch_size = 2+np.sum(2**np.arange(args.refine)) +2 *args.kernel_size
                        ]


# a mapping to reorder the indices of the original nonpadded patches
NONPAD_INDEX_MAPPING = [[],[],[],
                        [ 0, 
                          1,  2, 
                          4,  3,  6, 
                          5,  8,  7, 12, 
                         10,  9, 14, 13, 20, 
                         11, 16, 15, 22, 21, 29, 
                         18, 17, 24, 23, 31, 30, 36, 
                         19, 26, 25, 33, 32, 38, 37, 41,
                         28, 27, 35, 34, 40, 39, 43, 42, 44,
                        ],
                        [  0, 
                           1,   2, 
                           4,   3,   6, 
                           5,   8,   7,  12, 
                          10,   9,  14,  13,  20, 
                          11,  16,  15,  22,  21,  30, 
                          18,  17,  24,  23,  32,  31,  42, 
                          19,  26,  25,  34,  33,  44,  43,  56,
                          28,  27,  36,  35,  46,  45,  58,  57,  72,
                          29,  38,  37,  48,  47,  60,  59,  74,  73,  89,
                          40,  39,  50,  49,  62,  61,  76,  75,  91,  90, 104,
                          41,  52,  51,  64,  63,  78,  77,  93,  92, 106, 105, 117,
                          54,  53,  66,  65,  80,  79,  95,  94, 108, 107, 119, 118, 128,
                          55,  68,  67,  82,  81,  97,  96, 110, 109, 121, 120, 130, 129, 137,
                          70,  69,  84,  83,  99,  98, 112, 111, 123, 122, 132, 131, 139, 138, 144,
                          71,  86,  85, 101, 100, 114, 113, 125, 124, 134, 133, 141, 140, 146, 145, 149,
                          88,  87, 103, 102, 116, 115, 127, 126, 136, 135, 143, 142, 148, 147, 151, 150, 152,
                        ]]

# a mapping to reorder the indices of the padded patches 
# according to the structure described in the demo notebook
# back to its original order
PAD_REVERSE_INDEX_MAPPING = []
for rr in range(5):
    refinement = rr
    reverse_map = []
    if len(PAD_INDEX_MAPPING_KS2[refinement]):
        for ii in range(len(PAD_INDEX_MAPPING_KS2[refinement])-9):
            reverse_map += [PAD_INDEX_MAPPING_KS2[refinement].index(ii,3)]
        reverse_map[-1] += 1
    PAD_REVERSE_INDEX_MAPPING += [reverse_map]

    

# a mapping to reorder the indices of the nonpadded patches 
# according to the structure described in the demo notebook
# back to its original order
NONPAD_REVERSE_INDEX_MAPPING = []
for rr in range(5):
    refinement = rr
    reverse_map = []
    if len(NONPAD_INDEX_MAPPING[refinement]):
        for ii in range(len(NONPAD_INDEX_MAPPING[refinement])):
            reverse_map += [NONPAD_INDEX_MAPPING[refinement].index(ii)]
    NONPAD_REVERSE_INDEX_MAPPING += [reverse_map]

# the indices of the reordered padded patch that belong to the nonpadded patch for kernel size 2
NO_PADDING_INDICES = []
for refinement in range(5):
    counter = 12
    nopadind = []
    for rr in range(2+np.sum(2**np.arange(refinement))):
        for ii in range(rr+1):
            nopadind += [counter]
            counter += 1
        counter += 4
    NO_PADDING_INDICES += [nopadind]

    

# the indices of the original padded patch that belong to the nonpadded patch
NO_PADDING_INDICES_ORIGINAL = []
for refinement in range(5):
    nopadind = []
    if len(PAD_INDEX_MAPPING_KS2[refinement]):
        for ii in NO_PADDING_INDICES[refinement]:
            nopadind += [PAD_REVERSE_INDEX_MAPPING[refinement].index(ii)]
        nopadind.sort()
    NO_PADDING_INDICES_ORIGINAL += [nopadind]
        

### others
# some helper variables for nicely plotting the pooling and unpooling mechanism
PAD_LEVEL_3_RED_IDS = [12, 80, 88]
PAD_LEVEL_3_BLUE_IDS = [38, 42, 84]
PAD_LEVEL_3_GREEN_IDS = [23, 25, 40, 57, 59, 61, 63, 82, 86]

PAD_LEVEL_2_RED_IDS = [4, 22, 26]
PAD_LEVEL_2_BLUE_IDS = [11, 13, 24]
PAD_LEVEL_2_GREEN_IDS = [7, 8, 12, 16, 17, 18, 19, 23, 25]

PAD_LEVEL_1_RED_IDS = [0, 3, 5]
PAD_LEVEL_1_BLUE_IDS = [1, 2, 4]

NONPAD_LEVEL_3_RED_IDS = [0, 36, 44]
NONPAD_LEVEL_3_BLUE_IDS = [10, 14, 40]
NONPAD_LEVEL_3_GREEN_IDS = [3, 5, 12, 21, 23, 25, 27, 38, 42]

NONPAD_LEVEL_2_RED_IDS = [0, 10, 14]
NONPAD_LEVEL_2_BLUE_IDS = [3, 5, 12]
NONPAD_LEVEL_2_GREEN_IDS = [1, 2, 4, 6, 7, 8, 9, 11, 13]

NONPAD_LEVEL_1_RED_IDS = [0, 3, 5]
NONPAD_LEVEL_1_BLUE_IDS = [1, 2, 4]