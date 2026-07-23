"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : writing_mode.py
Author      : Anvitha K V
Description : Handles writing mode functionality
============================================================
"""


class WritingMode:
    """
    Manages letters, words and prediction history.
    """

    def __init__(self):

        self.current_word = ""
        self.current_letter = ""
        self.history = []

    # ------------------------------------------------------
    # Add Letter
    # ------------------------------------------------------

    def add_letter(self, letter):

        if not letter:
            return

        self.current_letter = letter
        self.current_word += letter
        self.history.append(letter)

    # ------------------------------------------------------
    # Delete Last Letter
    # ------------------------------------------------------

    def delete_last(self):

        if len(self.current_word) > 0:

            self.current_word = self.current_word[:-1]

        if len(self.history) > 0:

            self.history.pop()

    # ------------------------------------------------------
    # Add Space
    # ------------------------------------------------------

    def add_space(self):

        if len(self.current_word) > 0:

            if self.current_word[-1] != " ":

                self.current_word += " "

    # ------------------------------------------------------
    # Clear
    # ------------------------------------------------------

    def clear(self):

        self.current_word = ""
        self.current_letter = ""
        self.history.clear()

    # ------------------------------------------------------
    # Get History
    # ------------------------------------------------------

    def get_history(self):

        return self.history[-15:]

    # ------------------------------------------------------
    # Get Word
    # ------------------------------------------------------

    def get_word(self):

        return self.current_word

    # ------------------------------------------------------
    # Get Current Letter
    # ------------------------------------------------------

    def get_letter(self):

        return self.current_letter