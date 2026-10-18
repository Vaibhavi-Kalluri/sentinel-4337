// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract SentinelTarget {
    uint256 public value;
    mapping(address => uint256) public balances;

    event ValueChanged(
        address indexed caller,
        uint256 oldValue,
        uint256 newValue
    );

    event Deposit(
        address indexed from,
        uint256 amount
    );

    function setValue(uint256 newValue) external {
        uint256 oldValue = value;
        value = newValue;

        emit ValueChanged(msg.sender, oldValue, newValue);
    }

    function deposit() external payable {
        balances[msg.sender] += msg.value;

        emit Deposit(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");

        balances[msg.sender] -= amount;

        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");
    }
}
