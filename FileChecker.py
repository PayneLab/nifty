import os

class FileChecker:
    '''Ensures input files exist and are in the correct format.'''
    def __init__(self, input_files: list):
        self.input_files = input_files
        self.errors = []

    #insert functions here