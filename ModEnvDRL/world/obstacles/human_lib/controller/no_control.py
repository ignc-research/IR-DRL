# -*- coding: utf-8 -*-

from . import Controller


class NoControl(Controller):
    """No control implemenation"""

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def update(self, F, v_prev, omega_prev, v_cmd, omega_cmd):
        return (v_cmd, omega_cmd)
