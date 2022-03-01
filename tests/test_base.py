#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Test base.py

Project: sudoku-buddy
"""
import numpy as np
import pytest
from numpy.typing import ArrayLike

from sudoku_buddy.base import Sudoku


@pytest.mark.parametrize(
    ("cell", "row_neighbors", "column_neighbors", "box_neighbors", "expected_indices"),
    [
        [0, True, False, False, {1, 2, 3, 4, 5, 6, 7, 8}],
        [0, False, True, False, {9, 18, 27, 36, 45, 54, 63, 72}],
        [0, False, False, True, {1, 2, 9, 10, 11, 18, 19, 20}],
        [
            0,
            True,
            True,
            True,
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 27, 36, 45, 54, 63, 72},
        ],
        [1, True, False, False, {0, 2, 3, 4, 5, 6, 7, 8}],
        [80, True, False, False, {72, 73, 74, 75, 76, 77, 78, 79}],
    ],
)
def test_get_neighbor_indices(
    simple_sudoku: Sudoku,
    cell: int,
    row_neighbors: bool,
    column_neighbors: bool,
    box_neighbors: bool,
    expected_indices: set[int],
) -> None:
    assert (
        simple_sudoku._get_neighbor_indices(
            cell, row_neighbors, column_neighbors, box_neighbors
        )
        == expected_indices
    )


def test_Sudoku_constructor(simple_sudoku_buffer: ArrayLike) -> None:
    sudoku = Sudoku(simple_sudoku_buffer)
    assert sudoku._input_buffer is simple_sudoku_buffer
    assert sudoku._all_neighbors_indices.shape == (81, 20)
    assert sudoku._row_neighbors_indices.shape == (81, 8)
    assert sudoku._column_neighbors_indices.shape == (81, 8)
    assert sudoku._box_neighbors_indices.shape == (81, 8)


@pytest.fixture
def simple_sudoku(simple_sudoku_buffer: ArrayLike) -> Sudoku:
    return Sudoku(simple_sudoku_buffer)


@pytest.fixture
def simple_sudoku_buffer() -> ArrayLike:
    return [
        0,
        0,
        3,
        0,
        2,
        0,
        6,
        0,
        0,
        9,
        0,
        0,
        3,
        0,
        5,
        0,
        0,
        1,
        0,
        0,
        1,
        8,
        0,
        6,
        4,
        0,
        0,
        0,
        0,
        8,
        1,
        0,
        2,
        9,
        0,
        0,
        7,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        8,
        0,
        0,
        6,
        7,
        0,
        8,
        2,
        0,
        0,
        0,
        0,
        2,
        6,
        0,
        9,
        5,
        0,
        0,
        8,
        0,
        0,
        2,
        0,
        3,
        0,
        0,
        9,
        0,
        0,
        5,
        0,
        1,
        0,
        3,
        0,
        0,
    ]
