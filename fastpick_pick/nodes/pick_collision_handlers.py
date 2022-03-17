#!/usr/bin/env python3

'''
Muhammad Arshad 
17/03/2022
'''

from typing import Dict

import pandas as pd
import rospy
from moveit_msgs.msg import AllowedCollisionMatrix, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene


class CollisionManager:

    def __init__(self, namespace: str = "") -> None:
        self.ns = namespace
        self.get_planning_scene = rospy.ServiceProxy(
            f"get_planning_scene", GetPlanningScene
        )
        self.scene = rospy.Publisher(
            f"/move_group/monitored_planning_scene", PlanningScene, queue_size=0
        )

    def get_collision_matrix(self) -> AllowedCollisionMatrix:
        request = PlanningSceneComponents(
            components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        )
        return self.get_planning_scene(request).scene.allowed_collision_matrix

    def get_links(self, matrix: AllowedCollisionMatrix) -> Dict[str, int]:
        return {n: i for i, n in enumerate(matrix.entry_names)}

    def are_allowed(self, link_1: str, link_2: str) -> bool:
        matrix = self.get_collision_matrix()
        name_map = self.get_links(matrix)

        source_index = name_map[link_1]
        target_index = name_map[link_2]

        return bool(
            matrix.entry_values[source_index].enabled[target_index]
            and matrix.entry_values[target_index].enabled[source_index]
        )

    def toggle(self, link_1: str, link_2: str, allowed: bool) -> None:
        """
        Toggle collisions between two links in the URDF

        :param link_1:
        :param link_2:
        :param allowed:
        """
        matrix = self.get_collision_matrix()
        name_map = self.get_links(matrix)

        source_index = name_map[link_1]
        target_index = name_map[link_2]

        matrix.entry_values[source_index].enabled[target_index] = allowed
        matrix.entry_values[target_index].enabled[source_index] = allowed

        self.update_matrix(matrix)

    def update_matrix(self, matrix: AllowedCollisionMatrix) -> None:
        self.scene.publish(PlanningScene(is_diff=False, allowed_collision_matrix=matrix))

    @staticmethod
    def show_matrix(matrix: AllowedCollisionMatrix) -> object:
        # name_map = {i: n for i, n in enumerate(matrix.entry_names)}
        name_map = dict(enumerate(matrix.entry_names))
        rows = [["x", *matrix.entry_names]]
        for i, row in enumerate(matrix.entry_values):
            rows.append([name_map[i]] + row.enabled)
        pd.options.display.max_columns = None
        df = pd.DataFrame(rows)
        return df.rename(columns=df.iloc[0])


if __name__ == "__main__": 

    rospy.init_node("collision_manager_node")
    collisionManager = CollisionManager()
    colMat = collisionManager.get_collision_matrix()
    # print (collisionManager.show_matrix(colMat ) )  
    print (colMat)
    rospy.spin()