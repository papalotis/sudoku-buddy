#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Base file for sudoku solver implementation

Project: sudoku-buddy
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Sudoku:
    SUDOKU_SIZE = 9
    SUDOKU_SIZE_SQUARE = 81
    BOX_SIZE = 3
    EMPTY_VALUE = 0

    def _get_neighbor_indices(
        self,
        cell_index: int,
        row_neighbors: bool,
        column_neighbors: bool,
        box_neighbors: bool,
    ) -> set[int]:
        row, column = divmod(cell_index, self.SUDOKU_SIZE)
        box_row = row // self.BOX_SIZE
        box_column = column // self.BOX_SIZE

        neighbors: set[int] = set()

        for other_cell_index in range(self.SUDOKU_SIZE_SQUARE):
            if other_cell_index == cell_index:
                continue

            other_row, other_column = divmod(other_cell_index, self.SUDOKU_SIZE)
            if other_row == row and row_neighbors:
                neighbors.add(other_cell_index)

            if other_column == column and column_neighbors:
                neighbors.add(other_cell_index)

            other_box_row = other_row // self.BOX_SIZE
            other_box_column = other_column // self.BOX_SIZE
            if (
                other_box_row == box_row
                and other_box_column == box_column
                and box_neighbors
            ):
                neighbors.add(other_cell_index)

        return neighbors

    def _create_neighbors_indices_array(
        self, row_neighbors: bool, column_neighbors: bool, box_neighbors: bool
    ) -> NDArray[np.int_]:
        neighbros_lol = [
            list(
                self._get_neighbor_indices(
                    cell_index, row_neighbors, column_neighbors, box_neighbors
                )
            )
            for cell_index in range(self.SUDOKU_SIZE_SQUARE)
        ]

        return np.array(neighbros_lol, dtype=np.int_)

    def _handle_input_buffer(self) -> NDArray[np.int_]:
        return np.array(self._input_buffer, dtype=np.int_).flatten()

    def _create_initial_candidates_mask(self) -> NDArray[np.bool_]:
        mask: NDArray[np.bool_] = np.ones(
            (self.SUDOKU_SIZE_SQUARE, self.SUDOKU_SIZE), dtype=np.bool_
        )
        mask_value_is_not_empty = self._input_array != self.EMPTY_VALUE
        values_not_empty = self._input_array[mask_value_is_not_empty]
        # the index of a value is the same as the value itself minus 1 (because of
        # zero-based indexing)
        indices_to_set_to_true = values_not_empty - 1
        mask[mask_value_is_not_empty] = False
        mask[mask_value_is_not_empty, indices_to_set_to_true] = True
        return mask

    def __init__(self, input_buffer: ArrayLike):
        self._input_buffer = input_buffer
        self._input_array = self._handle_input_buffer()
        self._all_neighbors_indices = self._create_neighbors_indices_array(
            row_neighbors=True, column_neighbors=True, box_neighbors=True
        )
        self._row_neighbors_indices = self._create_neighbors_indices_array(
            row_neighbors=True, column_neighbors=False, box_neighbors=False
        )
        self._column_neighbors_indices = self._create_neighbors_indices_array(
            row_neighbors=False, column_neighbors=True, box_neighbors=False
        )
        self._box_neighbors_indices = self._create_neighbors_indices_array(
            row_neighbors=False, column_neighbors=False, box_neighbors=True
        )
        self._candidates_mask = self._create_initial_candidates_mask()
